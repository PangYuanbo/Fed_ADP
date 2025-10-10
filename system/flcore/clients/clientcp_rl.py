# ============================
# clientcp_rl.py
# 集成强化学习的FedCP客户端 - 自适应差分隐私阈值学习
# 基于原始clientcp.py修改，添加RL决策机制
# ============================

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, read_npz_data
import os
import shutil
import matplotlib.pyplot as plt

# 导入RL模块
from utils.simple_rl_dp import SimpleRLAgent, RLDPManager


class clientCP_RL:
    """
    集成强化学习的FedCP客户端
    在原始clientCP基础上添加自适应差分隐私阈值学习能力
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.alpha = args.alpha
        self.id = id
        self.dp = args.difference_privacy
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        print(f"Client {self.id} has {self.train_samples} training samples.")
        self.test_samples = test_samples

        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.noise = {}
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.round = 0

        # 日志目录设置
        log_root = "logs"
        self.param_diff = {}
        self.inital_pra = {}

        if self.dp:
            sub = f"{self.dataset}_gradient_log_dp_rl"  # 添加RL标识
        else:
            sub = f"{self.dataset}_gradient_log_rl"

        result_dir = os.path.join(log_root, sub)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)
        filename = f"gradient_log_client{self.id}_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}.txt"

        self.filepath = os.path.join(result_dir, filename)

        # 创建日志文件
        for suffix in ["testbefore", "test", "before", "after"]:
            with open(self.filepath + suffix, "a") as f:
                pass

        self.lamda = args.lamda

        in_dim = list(args.model.head.parameters())[0].shape[1]
        self.context = torch.rand(1, in_dim).to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # ============= RL 相关初始化 =============
        self.enable_rl = getattr(args, 'enable_rl_dp', False)  # 默认不启用RL
        self.rl_verbose = getattr(args, 'rl_verbose', False)  # 是否显示RL日志
        self.rl_agent = None
        self.rl_manager = None
        self.current_accuracy = 0.0
        self.current_mia_f_score = 0.5  # 默认中等风险

        if self.enable_rl and self.dp:  # 只有在启用DP时才启用RL
            try:
                # 初始化RL智能体
                self.rl_agent = SimpleRLAgent(
                    state_dim=3,
                    learning_rate=getattr(args, 'rl_learning_rate', 0.01),
                    epsilon=getattr(args, 'rl_epsilon', 0.1),
                    epsilon_decay=getattr(args, 'rl_epsilon_decay', 0.995),
                    device=self.device,
                    verbose=self.rl_verbose
                )

                # 初始化RL管理器
                self.rl_manager = RLDPManager(
                    agent=self.rl_agent,
                    update_interval=getattr(args, 'rl_update_interval', 10),
                    min_rounds_before_rl=getattr(args, 'rl_min_rounds', 20),
                    verbose=self.rl_verbose
                )

                # RL检查点文件路径
                self.rl_checkpoint_dir = f"rl_checkpoints/{self.dataset}"
                os.makedirs(self.rl_checkpoint_dir, exist_ok=True)
                self.rl_checkpoint_path = os.path.join(
                    self.rl_checkpoint_dir, f"client_{self.id}_rl_checkpoint.json"
                )

                # 尝试加载之前的检查点
                if os.path.exists(self.rl_checkpoint_path):
                    self.rl_manager.load_checkpoint(self.rl_checkpoint_path, verbose=self.rl_verbose)
                    if self.rl_verbose:
                        print(f"[Client {self.id}] Loaded RL checkpoint from {self.rl_checkpoint_path}")
                else:
                    if self.rl_verbose:
                        print(f"[Client {self.id}] Starting fresh RL training")

                if self.rl_verbose:
                    print(f"[Client {self.id}] RL-DP system initialized successfully")

            except Exception as e:
                print(f"[Client {self.id}] Failed to initialize RL system: {e}")
                self.enable_rl = False
                print(f"[Client {self.id}] Falling back to traditional DP")

    def compute_norm(self, param_dict):
        total_norm = 0.0
        for name, tensor in param_dict.items():
            total_norm += tensor.norm().item() ** 2
        return total_norm ** 0.5

    def get_module_grad_norm(self, model) -> float:
        """计算给定模块的所有参数梯度的 L2 范数并返回"""
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        return total_norm_sq ** 0.5

    def get_module_gradient_norm(self, dataloader, filename: str):
        with torch.enable_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                break
            grad_norm_head = self.get_module_grad_norm(self.model.head)
            grad_norm_feat = self.get_module_grad_norm(self.model.feature_extractor)

            record_dict = {
                "round": self.round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }
            with open(self.filepath + filename, "a") as f:
                f.write(str(record_dict) + "\n")

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, alpha=self.alpha)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, alpha=self.alpha)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def set_parameters(self, feature_extractor):
        for new_param, old_param in zip(feature_extractor.parameters(), self.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()

    def set_head_g(self, head):
        headw_ps = []
        for name, mat in self.model.model.head.named_parameters():
            if 'weight' in name:
                headw_ps.append(mat.data)
        headw_p = headw_ps[-1]
        for mat in headw_ps[-2::-1]:
            headw_p = torch.matmul(headw_p, mat)
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)

    def train_metrics(self):
        """计算训练集上的准确率"""
        trainloader = self.load_train_data()
        self.model.eval()

        train_acc = 0
        train_num = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]

        return train_acc, train_num

    def test_metrics_before(self):
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        # 更新当前准确率用于RL
        self.current_accuracy = test_acc / test_num if test_num > 0 else 0.0

        return test_acc, test_num, auc

    def test_metrics_after(self):
        testloader = self.load_test_data()
        self.model.train()
        self.pm_test = []
        self.get_module_gradient_norm(dataloader=testloader, filename="test")
        return

    def update_mia_score(self, mia_f_score: float):
        """更新MIA F-score，由服务器端调用"""
        self.current_mia_f_score = mia_f_score

    def get_rl_adaptive_thresholds(self) -> tuple:
        """
        使用RL获取自适应阈值
        返回 (threshold_high, threshold_low) 或 None 表示使用默认值
        """
        if not self.enable_rl or not self.rl_manager:
            return None

        try:
            thresholds = self.rl_manager.get_thresholds(
                round_num=self.round,
                accuracy=self.current_accuracy,
                mia_f_score=self.current_mia_f_score
            )
            return thresholds
        except Exception as e:
            print(f"[Client {self.id}] RL threshold selection failed: {e}")
            return None

    def save_rl_checkpoint(self):
        """保存RL检查点"""
        if self.enable_rl and self.rl_manager:
            try:
                self.rl_manager.save_checkpoint(self.rl_checkpoint_path)
                if self.round % 50 == 0:  # 每50轮打印一次统计
                    print(f"[Client {self.id}] RL Statistics:")
                    self.rl_agent.print_statistics()
            except Exception as e:
                print(f"[Client {self.id}] Failed to save RL checkpoint: {e}")

    def train_cs_model(self, round, args):
        testloader = self.load_test_data()
        trainloader = self.load_train_data()

        self.get_module_gradient_norm(dataloader=testloader, filename="test_before")
        self.model.train()

        # 模型的真实训练部分
        for _ in range(self.local_steps):
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.inital_pra = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        self.inital_pra_dp = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        for _ in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                break

            grad_norm_head = self.get_module_grad_norm(self.model.head)
            grad_norm_feat = self.get_module_grad_norm(self.model.feature_extractor)

            record_dict = {
                "round": self.round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }
            with open(self.filepath + "train_before", "a") as f:
                f.write(str(record_dict) + "\n")

            for _ in range(self.local_steps):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.opt.zero_grad()
                    loss.backward(retain_graph=True)
                    break

            # Hessian分析 (保持原有逻辑)
            if self.round % 10 == 0:
                diag_H = hessian_diag_hutchinson(self.model, loss, num_samples=8)
                named_params = list(self.model.named_parameters())
                hess_dict = {name: h for (name, _), h in zip(named_params, diag_H)}
                self.hess_mask = {}

                for name, h in hess_dict.items():
                    h_abs = h.abs().flatten()
                    if h_abs.numel() == 0:
                        continue
                    thresh_1 = torch.quantile(h_abs, 0.60).item()
                    thresh_2 = torch.quantile(h_abs, 0.40).item()
                    self.hess_mask[name] = torch.logical_or(h.abs() >= thresh_1, h.abs() <= thresh_2)
            else:
                if not hasattr(self, "hess_mask"):
                    self.hess_mask = {}

        # 制作test threshold_test
        with torch.enable_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                break
            self.param_diff_test = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    grad = p.grad.data
                    self.param_diff_test[name] = -self.learning_rate * grad

        # ============= RL自适应差分隐私部分 =============
        clip_value = 0.005
        epsilon = 0.8
        delta = 1e-5

        if self.dp:
            param_diff = {}
            modules = {'feature_extractor': self.model.feature_extractor}

            for module_name, module in modules.items():
                for name, param in module.named_parameters():
                    full_name = f"{module_name}.{name}".lstrip('.')
                    param_diff[full_name] = (param - self.inital_pra_dp[full_name]).detach()

            # 复原参数
            for name, param in self.model.named_parameters():
                param.data = self.inital_pra_dp[name].clone().detach()

            param_public_diff = {}
            for module_name, module in modules.items():
                for name, param in module.named_parameters():
                    full_name = f"{module_name}.{name}".lstrip('.')
                    param_public_diff[full_name] = (param - self.inital_pra_dp[full_name]).detach()

            for full_name, diff in param_diff.items():
                norm_train = torch.norm(diff.abs())
                norm_test = torch.norm(self.param_diff_test[full_name].abs())

                # ============= 使用RL自适应阈值 =============
                rl_thresholds = self.get_rl_adaptive_thresholds()
                if rl_thresholds is not None:
                    threshold_1_percentile, threshold_2_percentile = rl_thresholds
                    if self.rl_verbose:
                        print(f"[Client {self.id}] Round {self.round}: Using RL thresholds ({threshold_1_percentile:.2f}, {threshold_2_percentile:.2f})")
                else:
                    # 使用默认阈值
                    threshold_1_percentile, threshold_2_percentile = 0.6, 0.4
                    if self.rl_verbose:
                        print(f"[Client {self.id}] Round {self.round}: Using default thresholds ({threshold_1_percentile:.2f}, {threshold_2_percentile:.2f})")

                # 应用RL选择的阈值
                threshold_1 = torch.quantile(diff.abs().view(-1), threshold_1_percentile)
                threshold_2 = torch.quantile(diff.abs().view(-1), threshold_2_percentile)

                core_mask = torch.logical_or((diff.abs() >= threshold_1), (diff.abs() <= threshold_2))
                q = torch.quantile(param_public_diff[full_name].abs().view(-1), 0.5)

                # 记录clip value
                result_dir = "clip_value"
                os.makedirs(result_dir, exist_ok=True)
                filename = f"results_{self.dataset}_{self.id}.txt"
                file_path = os.path.join(result_dir, filename)
                with open(file_path, "a") as f:
                    f.write(f"Round {self.round}: clip_value = {clip_value}, RL_thresholds = {rl_thresholds}\n")

                # 应用Hessian掩码
                if full_name in self.hess_mask:
                    core_mask &= self.hess_mask[full_name]
                    total_params = core_mask.numel()
                    selected_params = core_mask.sum().item()
                    percentage = selected_params / total_params * 100

                masked_diff = diff[core_mask]

                # 裁剪
                norm = torch.norm(diff)
                if norm > clip_value:
                    diff = diff / norm * clip_value

                # 加噪
                noise_std_estimate = (clip_value / epsilon) * torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.25 / delta))
                )
                noise = torch.normal(mean=0, std=noise_std_estimate, size=masked_diff.shape).to(diff.device)
                masked_diff = masked_diff + noise
                norm_masked_noisy = torch.norm(masked_diff.abs())

                # 写回
                diff[core_mask] = masked_diff
                param_diff[full_name] = diff

                # 记录噪声
                self.noise[full_name] = torch.zeros_like(diff)
                self.noise[full_name][core_mask] = noise

            # 应用参数更新
            for name, param in self.model.named_parameters():
                if name in param_diff:
                    param.data = self.inital_pra_dp[name] + param_diff[name]
                else:
                    param.data = self.inital_pra_dp[name]

        self.get_module_gradient_norm(dataloader=trainloader, filename="train_after")

        # 保存RL检查点
        self.save_rl_checkpoint()


def hessian_diag_hutchinson(model, loss, params=None, num_samples=8):
    """Hutchinson对角Hessian估计 (保持原有实现)"""
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # 一阶梯度（需要保留计算图）
    grads = torch.autograd.grad(loss, params, create_graph=True)

    diag_est = [torch.zeros_like(p) for p in params]
    for _ in range(num_samples):
        vs = [torch.randint_like(p, 2, dtype=torch.float32) * 2 - 1   # Rademacher ±1
              for p in params]
        Hv = torch.autograd.grad(grads, params, grad_outputs=vs,
                                 retain_graph=True)
        for d, hv in zip(diag_est, Hv):
            d += hv.pow(2)        # (Hv)⊙v，元素平方即可
    return [d / num_samples for d in diag_est]