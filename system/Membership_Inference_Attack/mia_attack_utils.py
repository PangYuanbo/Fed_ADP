# ============================
# mia_attack_utils.py
# ============================

import math
import torch
import torch.nn.functional as F
import numpy as np

from utils.attack_feature_config import (
    DEFAULT_ATTACK_FEATURES,
    FEATURE_SPECS,
    normalize_attack_features,
)

# 获取模型输出 + 梯度信息
def get_model_outputs_labels_and_grads(model, dataloader, device):
    model.eval()
    outputs_list, labels_list = [], []
    head_grads, feat_grads = {}, {}

    for name, param in model.head.named_parameters():
        head_grads[name] = []
    for name, param in model.feature_extractor.named_parameters():
        feat_grads[name] = []

    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()

        softmax = torch.nn.functional.softmax(logits.detach().cpu(), dim=1)
        outputs_list.append(softmax.numpy())

        labels_list.append(batch_y.detach().cpu().numpy())

        for name, param in model.head.named_parameters():
            grad = param.grad.detach().cpu().numpy().copy() if param.grad is not None else None
            head_grads[name].append(grad)

        for name, param in model.feature_extractor.named_parameters():
            grad = param.grad.detach().cpu().numpy().copy() if param.grad is not None else None
            feat_grads[name].append(grad)

        # 🔑 关键修复: 立即删除计算图，释放内存
        del loss, logits, softmax, batch_x, batch_y
        model.zero_grad()  # 清理梯度

        # 每10个batch强制清理GPU缓存
        if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    outputs = np.concatenate(outputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Validate data consistency
    expected_len = len(outputs)
    assert len(labels) == expected_len, "Mismatch between outputs and labels"
    for grad_dict in [head_grads, feat_grads]:
        for k, v in grad_dict.items():
            assert len(v) == expected_len, f"Mismatch in gradient count for {k}"

    return outputs, labels, head_grads, feat_grads


def _stack_grad_list(grad_list, name):
    grads = [torch.tensor(g, dtype=torch.float32) for g in grad_list if g is not None]
    if not grads:
        raise ValueError(f"No gradients collected for parameter '{name}'.")
    return torch.stack(grads, dim=0)


def _reshape_tensor(tensor):
    batch = tensor.shape[0]
    flat = tensor.view(batch, -1)
    side = math.ceil(flat.shape[1] ** 0.5)
    pad = side * side - flat.shape[1]
    if pad > 0:
        flat = F.pad(flat, (0, pad), value=0)
    return flat.view(batch, 1, side, side)


def _softmax_tensor(outputs):
    if isinstance(outputs, torch.Tensor):
        return F.softmax(outputs, dim=1).clone().detach().float()
    return F.softmax(torch.from_numpy(outputs), dim=1).float()


def _collect_feature_tensors(outputs, head_grads, feat_grads, feature_order):
    tensors = {}
    if 'softmax' in feature_order:
        tensors['softmax'] = _softmax_tensor(outputs)
    for name in feature_order:
        spec = FEATURE_SPECS[name]
        if spec['grad_source'] == 'softmax':
            continue
        source = feat_grads if spec['grad_source'] == 'feature' else head_grads
        grad_list = source[spec['grad_key']]
        tensors[name] = _reshape_tensor(_stack_grad_list(grad_list, spec['grad_key']))
    return tensors

# 白盒攻击输入预处理（reshape）
def prepare_attack_model_inputs(outputs,
                                head_grads,
                                feat_grads,
                                enabled_features=None,
                                return_dict=False):
    feature_order = normalize_attack_features(enabled_features) if return_dict else DEFAULT_ATTACK_FEATURES
    tensors = _collect_feature_tensors(outputs, head_grads, feat_grads, feature_order)
    if return_dict:
        return {name: tensors[name] for name in feature_order}
    return (
        tensors['conv1'],
        tensors['conv2'],
        tensors['fc1'],
        tensors['fc'],
        tensors['softmax'],
    )


# ============= GPU优化版本 (5090大显存专用) =============

def get_model_outputs_labels_and_grads_gpu(model, dataloader, device):
    """
    GPU优化版本（流式处理）：逐batch生成，避免累积所有梯度

    这是一个生成器函数，每次yield一个batch的结果，避免内存爆炸

    Yields:
        tuple: (batch_outputs, batch_labels, batch_head_grads, batch_feat_grads)
    """
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()

        # 计算softmax输出（保持在GPU上）
        softmax = torch.nn.functional.softmax(logits.detach(), dim=1)

        # 获取当前batch的梯度（保持在GPU上）
        batch_head_grads = {}
        for name, param in model.head.named_parameters():
            if param.grad is not None:
                batch_head_grads[name] = param.grad.detach().clone()

        batch_feat_grads = {}
        for name, param in model.feature_extractor.named_parameters():
            if param.grad is not None:
                batch_feat_grads[name] = param.grad.detach().clone()

        # 清理计算图
        del loss, logits
        model.zero_grad()

        # 🔑 yield当前batch的结果，不累积
        yield softmax, batch_y.detach(), batch_head_grads, batch_feat_grads

        # 清理当前batch数据
        del batch_x, batch_y, softmax, batch_head_grads, batch_feat_grads


def prepare_attack_model_inputs_gpu(outputs, head_grads, feat_grads, device):
    """
    GPU优化版本：所有操作在GPU上完成（用于批量数据）

    Args:
        outputs: GPU tensor
        head_grads: dict of GPU tensor lists (for multiple samples)
        feat_grads: dict of GPU tensor lists (for multiple samples)
        device: GPU device
    """
    # 已经在GPU上，直接使用
    softmax_out = F.softmax(outputs, dim=1).to(device)

    def stack_grads(grad_list):
        grads = [g.to(device) for g in grad_list if g is not None]
        return torch.stack(grads, dim=0)

    grad_conv1 = stack_grads(feat_grads['conv1.0.weight'])
    grad_conv2 = stack_grads(feat_grads['conv2.0.weight'])
    grad_fc1   = stack_grads(feat_grads['fc1.0.weight'])
    grad_fc = stack_grads(head_grads['weight'])

    def reshape_tensor(tensor):
        B = tensor.shape[0]
        flat = tensor.view(B, -1)
        side = int(flat.shape[1] ** 0.5)
        if side * side != flat.shape[1]:
            pad = side * side - flat.shape[1]
            flat = F.pad(flat, (0, pad), value=0)
        return flat.view(B, 1, side, side)

    # 返回GPU tensors
    return (reshape_tensor(grad_conv1),
            reshape_tensor(grad_conv2),
            reshape_tensor(grad_fc1),
            reshape_tensor(grad_fc),
            softmax_out)


def prepare_attack_model_inputs_single_gpu(outputs, head_grads, feat_grads, device):
    """
    为单个batch准备攻击模型输入（GPU流式处理版本）

    Args:
        outputs: [batch_size, num_classes] - softmax输出（GPU tensor）
        head_grads: dict of tensors - 单个batch的head梯度（每个值是tensor，不是列表）
        feat_grads: dict of tensors - 单个batch的feature梯度（每个值是tensor，不是列表）
        device: GPU device

    Returns:
        tuple: 准备好的攻击模型输入（所有在GPU上）
    """
    # softmax已经在GPU上
    softmax_out = F.softmax(outputs, dim=1).to(device)

    # 🔑 关键：直接使用tensor，不需要stack（因为已经是单个batch的tensor）
    grad_conv1 = feat_grads['conv1.0.weight'].to(device)
    grad_conv2 = feat_grads['conv2.0.weight'].to(device)
    grad_fc1 = feat_grads['fc1.0.weight'].to(device)
    grad_fc = head_grads['weight'].to(device)

    def reshape_tensor(tensor, name=''):
        # 🔑 匹配训练时的reshape方式
        # 训练时: stack后的梯度 (N_samples, param_dims...) → view(N, -1) → reshape(N, 1, H, H)
        # 推理时: 单个梯度 (param_dims...) → 我们把param_dim0当作N来处理

        param_dim0 = tensor.shape[0]  # 参数的第一维 (out_channels等)
        flat = tensor.view(param_dim0, -1)  # 保留第一维，展平其余维度

        side = int(flat.shape[1] ** 0.5)
        if side * side != flat.shape[1]:
            pad = side * side - flat.shape[1]
            flat = F.pad(flat, (0, pad), value=0)

        # Reshape: (param_dim0, 1, side, side)
        reshaped = flat.view(param_dim0, 1, side, side)

        # 池化到单样本: (param_dim0, 1, side, side) → (1, 1, side, side)
        # 使用mean pooling沿第一维聚合
        result = reshaped.mean(dim=0, keepdim=True)

        return result

    # 返回GPU tensors
    return (reshape_tensor(grad_conv1, 'conv1'),
            reshape_tensor(grad_conv2, 'conv2'),
            reshape_tensor(grad_fc1, 'fc1'),
            reshape_tensor(grad_fc, 'fc'),
            softmax_out)
