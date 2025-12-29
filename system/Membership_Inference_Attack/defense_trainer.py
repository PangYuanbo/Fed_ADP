import torch
import torch.nn.functional as F

from utils.attack_feature_config import FEATURE_SPECS, normalize_attack_features


def _collect_needed_keys(attack_features):
    feature_keys = set()
    head_keys = set()
    include_softmax = False
    for name in attack_features:
        spec = FEATURE_SPECS[name]
        if spec['grad_source'] == 'feature':
            feature_keys.add(spec['grad_key'])
        elif spec['grad_source'] == 'head':
            head_keys.add(spec['grad_key'])
        elif spec['grad_source'] == 'softmax':
            include_softmax = True
    return feature_keys, head_keys, include_softmax


def _reshape_grad_tensor(tensor):
    if tensor is None:
        return None
    flat = tensor.view(1, -1)
    side = int(flat.shape[1] ** 0.5)
    if side * side < flat.shape[1]:
        side += 1
    pad = side * side - flat.shape[1]
    if pad > 0:
        flat = F.pad(flat, (0, pad), value=0.0)
    return flat.view(1, 1, side, side)


def _compute_attack_inputs_with_graph(model, batch_x, batch_y, attack_features):
    normalized = normalize_attack_features(attack_features)
    feature_keys, head_keys, include_softmax = _collect_needed_keys(normalized)

    logits = model(batch_x)
    loss = F.cross_entropy(logits, batch_y)

    defense_layers = getattr(model.feature_extractor, "_defense_layers", None)
    if not defense_layers:
        raise RuntimeError("Defense layers are not installed on the feature extractor.")
    for layer in defense_layers:
        for p in layer.parameters():
            p.requires_grad_(True)

    feat_params, feat_names = [], []
    toggled_feat = []
    for name, param in model.feature_extractor.named_parameters():
        if name in feature_keys:
            if not param.requires_grad:
                param.requires_grad_(True)
                toggled_feat.append(param)
            feat_params.append(param)
            feat_names.append(name)

    head_params, head_names = [], []
    toggled_head = []
    for name, param in model.head.named_parameters():
        if name in head_keys:
            if not param.requires_grad:
                param.requires_grad_(True)
                toggled_head.append(param)
            head_params.append(param)
            head_names.append(name)

    feat_grads = {}
    if feat_params:
        grads = torch.autograd.grad(
            loss,
            feat_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        for key, grad in zip(feat_names, grads):
            feat_grads[key] = grad

    head_grads = {}
    if head_params:
        grads = torch.autograd.grad(
            loss,
            head_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        for key, grad in zip(head_names, grads):
            head_grads[key] = grad

    for param in toggled_feat:
        param.requires_grad_(False)
    for param in toggled_head:
        param.requires_grad_(False)

    feature_tensors = {}
    for name in normalized:
        spec = FEATURE_SPECS[name]
        if spec['grad_source'] == 'softmax':
            feature_tensors[name] = F.softmax(logits, dim=1)
        elif spec['grad_source'] == 'feature':
            grad = feat_grads.get(spec['grad_key'])
            tensor = _reshape_grad_tensor(grad)
        else:
            grad = head_grads.get(spec['grad_key'])
            tensor = _reshape_grad_tensor(grad)
        if spec['grad_source'] != 'softmax':
            if tensor is None:
                tensor = torch.zeros(1, 1, 1, 1, device=logits.device, requires_grad=True)
            else:
                tensor = tensor.requires_grad_()
            feature_tensors[name] = tensor

    return feature_tensors


def _binary_f1(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _eval_model_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    if total == 0:
        return 0.0
    return correct / total


def train_defense_layers(model,
                         attack_model,
                         member_loader,
                         device,
                         attack_features,
                         cfg,
                         eval_loader=None):
    if not cfg or not cfg.get('enabled'):
        return
    defense_params = model.feature_extractor.defense_parameters()
    if not defense_params:
        print("[Defense] No defense layers installed; skipping.")
        return

    lr = cfg.get('lr', 1e-3)
    epochs = cfg.get('epochs', 1)
    max_batches = cfg.get('max_batches')
    optimizer = torch.optim.Adam(defense_params, lr=lr)
    attack_model.eval()
    for p in attack_model.parameters():
        p.requires_grad = False
    model.feature_extractor.freeze_base_parameters()
    for p in model.head.parameters():
        p.requires_grad = False

    step = 0
    last_loss = None
    for epoch in range(epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(member_loader):
            if max_batches is not None and step >= max_batches:
                break
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            model.zero_grad()

            feature_inputs = _compute_attack_inputs_with_graph(
                model,
                batch_x,
                batch_y,
                attack_features,
            )
            attack_out = attack_model(feature_inputs)
            target = torch.zeros_like(attack_out)
            loss = F.binary_cross_entropy_with_logits(attack_out, target)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

            step += 1
        if max_batches is not None and step >= max_batches:
            break
        if last_loss is not None:
            preds = torch.sigmoid(attack_out).detach()
            f1 = _binary_f1(preds, torch.zeros_like(preds))
            acc_note = ""
            if eval_loader is not None:
                acc = _eval_model_accuracy(model, eval_loader, device)
                acc_note = f" | Acc(after defense)={acc:.4f}"
            print(f"[Defense] Epoch {epoch+1}/{epochs} Loss={last_loss:.4f} AttackF1~{f1:.4f}{acc_note}")
    for p in attack_model.parameters():
        p.requires_grad = True
    model.feature_extractor.unfreeze_all_parameters()
    for p in model.head.parameters():
        p.requires_grad = True
