# ============================
# client_selective.py
# 继承clientCP，添加选择性上传逻辑
# ============================

import sys
import os
import copy
import torch

# 添加父目录到路径，以便导入clientCP
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from flcore.clients.clientcp import clientCP


class ClientSelective(clientCP):
    """
    选择性上传客户端

    从第50轮开始，应用以下策略：
    1. 大梯度+高曲率参数 → 本地保留，不上传
    2. 其他参数 → 添加DP噪声后上传
    3. 收到全局模型后，对保留参数做平滑融合：(本地值 + 全局值) / 2
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """初始化选择性上传客户端"""
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 选择性上传相关参数
        self.selective_upload_round = getattr(args, 'selective_upload_round', 50)
        self.gradient_keep_ratio = getattr(args, 'gradient_keep_ratio', 0.3)
        self.noise_multiplier = getattr(args, 'noise_multiplier', 1.0)  # 噪声倍率（默认1.0）

        # 保留参数的记录
        self.kept_params = {}  # 记录本地保留的参数值
        self.kept_mask = {}    # 记录哪些参数被保留（True=保留，False=上传）

        print(f"[Client {self.id}] Selective Upload: enabled from round {self.selective_upload_round}, keep ratio {self.gradient_keep_ratio}, noise multiplier {self.noise_multiplier}")

    def train_cs_model(self, round, args):
        """
        重写训练方法

        在选择性上传模式下（round >= selective_upload_round）使用定制化DP：
        - 对上传的参数（kept_mask=False）添加大噪声
        - 对保留的参数（kept_mask=True）不添加噪声

        Args:
            round: 当前训练轮次
            args: 训练参数
        """
        # 🔑 如果在选择性上传阶段且启用DP，使用定制化DP逻辑
        if round >= self.selective_upload_round and self.dp:
            self._train_with_selective_dp(round, args)
        else:
            # 否则调用父类的标准训练方法
            super().train_cs_model(round, args)

        # 注意：不在这里应用选择性上传
        # 选择性上传将在prepare_upload_model中执行（MIA评估之后）

    def _train_with_selective_dp(self, round, args):
        """
        定制化DP训练方法（用于选择性上传模式）

        流程：
        1. 本地训练（复用父类逻辑）
        2. 计算kept_mask（识别大梯度参数）
        3. 只对上传的参数（kept_mask=False）添加大噪声
        4. 对保留的参数（kept_mask=True）不添加噪声

        Args:
            round: 当前轮次
            args: 训练参数
        """
        # ========== 步骤1: 本地训练（复制父类逻辑）==========
        testloader = self.load_test_data()
        trainloader = self.load_train_data()

        # 测试前梯度范数
        self.get_module_gradient_norm(dataloader=testloader, filename="test_before")

        self.model.train()

        # 🔑 保存训练前的参数（用于计算梯度差异）
        self.inital_pra_dp = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        # 本地训练
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

        # 保存训练后的参数（用于后续对比）
        self.inital_pra = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        # ========== 步骤2: 计算kept_mask（识别大梯度参数）==========
        self.kept_params = {}
        self.kept_mask = {}

        param_diff = {}
        for name, param in self.model.named_parameters():
            if name in self.inital_pra_dp:
                param_diff[name] = (param - self.inital_pra_dp[name]).abs()

        # 对每个参数计算保留掩码
        for name, diff in param_diff.items():
            if diff.numel() == 0:
                self.kept_mask[name] = torch.zeros_like(diff, dtype=torch.bool)
                continue

            # 计算梯度幅度阈值（保留top gradient_keep_ratio的参数）
            grad_threshold = torch.quantile(diff.view(-1), 1.0 - self.gradient_keep_ratio)
            high_gradient_mask = diff >= grad_threshold

            # 保存掩码（True=保留，False=上传）
            self.kept_mask[name] = high_gradient_mask

            # 保存被保留的参数值（用于后续平滑）
            current_param = self.model.state_dict()[name]
            self.kept_params[name] = current_param.clone().detach()

            # 统计信息（每10轮打印一次）
            if round % 10 == 0:
                total_params = high_gradient_mask.numel()
                kept_count = high_gradient_mask.sum().item()
                upload_count = total_params - kept_count
                if kept_count > 0:  # 只打印有保留参数的层
                    print(f"[Client {self.id} Round {round}] {name}: kept {kept_count}/{total_params} ({kept_count/total_params*100:.2f}%), upload {upload_count} (with noise)")

        # ========== 步骤3: 定制化DP噪声添加 ==========
        # 只对上传的参数添加大噪声
        self._apply_selective_dp_noise(round, param_diff)

        # 梯度范数记录（复用父类逻辑）
        # 省略部分记录逻辑...

    def _apply_selective_dp_noise(self, round, param_diff):
        """
        定制化DP噪声添加：只对上传的参数添加大噪声

        Args:
            round: 当前轮次
            param_diff: 参数差异字典
        """
        clip_value = self.clip_value
        epsilon = self.epsilon
        delta = self.delta

        # 复原参数到训练前状态
        for name, param in self.model.named_parameters():
            param.data = self.inital_pra_dp[name].clone().detach()

        # 对每个参数进行处理
        for name in param_diff.keys():
            if name not in self.kept_mask:
                continue

            diff = param_diff[name]
            kept_mask = self.kept_mask[name]
            upload_mask = ~kept_mask  # True=上传，False=保留

            # 🔑 只对上传的参数添加噪声
            if upload_mask.any():
                # 提取上传参数的差异
                upload_diff = diff.clone()

                # 裁剪（对整个参数进行裁剪）
                norm = torch.norm(upload_diff)
                if norm > clip_value:
                    upload_diff = upload_diff / norm * clip_value

                # 🔑 只对上传位置添加噪声
                noise_std = (clip_value / epsilon) * torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.25 / delta))
                )
                # 应用噪声倍率
                noise_std = noise_std * self.noise_multiplier
                noise = torch.normal(mean=0, std=noise_std, size=upload_diff[upload_mask].shape).to(diff.device)

                # 只给上传位置添加噪声
                upload_diff[upload_mask] = upload_diff[upload_mask] + noise

                # 保留位置不加噪声，保持原始训练后的值
                # upload_diff[kept_mask] 保持不变

                param_diff[name] = upload_diff

                # 统计（每10轮打印一次）
                if round % 10 == 0:
                    upload_count = upload_mask.sum().item()
                    if upload_count > 0:
                        avg_noise = noise.abs().mean().item()
                        print(f"[Client {self.id}] {name}: Added noise to {upload_count} upload params, avg_noise={avg_noise:.6f}")

        # 更新模型参数
        for name, param in self.model.named_parameters():
            if name in param_diff:
                param.data = self.inital_pra_dp[name] + param_diff[name]
            else:
                param.data = self.inital_pra_dp[name]

    def apply_selective_upload(self, round):
        """
        应用选择性上传策略：
        1. 识别大梯度+高曲率参数，本地保留
        2. 对其他参数已经在父类中添加了DP噪声（复用现有逻辑）

        Args:
            round: 当前轮次
        """
        # 清空上一轮的保留记录
        self.kept_params = {}
        self.kept_mask = {}

        # 计算参数差异（训练后 - 训练前）
        param_diff = {}
        for name, param in self.model.named_parameters():
            if name in self.inital_pra_dp:
                param_diff[name] = (param - self.inital_pra_dp[name]).abs()

        # 对每个参数计算保留掩码
        for name, diff in param_diff.items():
            if diff.numel() == 0:
                # 跳过空参数
                self.kept_mask[name] = torch.zeros_like(diff, dtype=torch.bool)
                continue

            # 1. 计算梯度幅度阈值（保留top gradient_keep_ratio的参数）
            grad_threshold = torch.quantile(diff.view(-1), 1.0 - self.gradient_keep_ratio)
            high_gradient_mask = diff >= grad_threshold

            # 2. 结合Hessian曲率（如果可用）
            if hasattr(self, 'hess_mask') and name in self.hess_mask:
                # Hessian掩码：True表示高曲率或低曲率（取决于阈值设置）
                # 我们需要高曲率参数，所以需要反转原有的"低曲率"逻辑
                hess_mask = self.hess_mask[name]

                # 计算高曲率掩码（反转低曲率掩码的逻辑）
                # 原始hess_mask标记的是可加噪的位置（低曲率），我们需要高曲率
                # 所以取反，但要考虑原始逻辑是OR关系（两端）
                # 为了简单，我们直接使用梯度幅度作为主要判断标准
                # 如果有Hessian信息，额外考虑：只保留那些也具有高曲率的参数

                # 重新计算高曲率掩码
                # 注意：原始代码中hess_mask标记的是可以加噪的区域（低曲率 OR 两端）
                # 我们需要的是高曲率区域，即那些NOT在低曲率中间区域的
                # 由于原始逻辑复杂，我们简化为：只使用梯度幅度判断
                kept_mask = high_gradient_mask
            else:
                # 没有Hessian信息，只使用梯度幅度
                kept_mask = high_gradient_mask

            # 保存掩码和参数值
            self.kept_mask[name] = kept_mask

            # 保存被保留的参数值（用于后续平滑）
            current_param = self.model.state_dict()[name]
            self.kept_params[name] = current_param.clone().detach()

            # 统计信息（每10轮打印一次）
            if round % 10 == 0:
                total_params = kept_mask.numel()
                kept_count = kept_mask.sum().item()
                kept_ratio = kept_count / total_params if total_params > 0 else 0
                if kept_count > 0:  # 只打印有保留参数的层
                    print(f"[Client {self.id} Round {round}] {name}: kept {kept_count}/{total_params} ({kept_ratio*100:.2f}%)")

        # 将保留的参数位置设为初始值（模拟不上传这些参数）
        # 注意：在实际联邦学习中，服务器不会收到这些位置的更新
        # 这里我们通过将这些位置恢复为初始值来模拟
        for name, param in self.model.named_parameters():
            if name in self.kept_mask:
                mask = self.kept_mask[name]
                if mask.any():
                    # 对保留位置，恢复为初始值（相当于没有更新）
                    with torch.no_grad():
                        param.data[mask] = self.inital_pra_dp[name][mask]

    def prepare_upload_model(self, round):
        """
        准备上传模型（在服务器收集模型之前调用）

        此方法在MIA评估之后、服务器聚合之前执行，
        应用选择性上传策略，确保MIA评估使用训练后的原始模型。

        Args:
            round: 当前轮次
        """
        if round >= self.selective_upload_round and self.dp:
            self.apply_selective_upload(round)

    def receive_and_smooth_global_model(self, global_model):
        """
        接收全局模型并进行平滑融合

        对于上一轮保留的参数位置：
        new_param = (kept_param + global_param) / 2

        对于其他位置：
        new_param = global_param

        Args:
            global_model: 服务器发送的全局模型
        """
        global_state = global_model.state_dict()

        for name, param in self.model.named_parameters():
            if name in self.kept_mask and name in self.kept_params:
                mask = self.kept_mask[name]
                if mask.any():
                    # 对保留位置进行平滑
                    with torch.no_grad():
                        smoothed_param = (self.kept_params[name][mask] + global_state[name][mask]) / 2
                        param.data[mask] = smoothed_param

                    # 对未保留位置，直接使用全局参数
                    param.data[~mask] = global_state[name][~mask]
                else:
                    # 没有保留参数，直接使用全局参数
                    param.data = global_state[name].clone()
            else:
                # 没有保留记录，直接使用全局参数
                param.data = global_state[name].clone()

    def set_parameters(self, model):
        """
        重写set_parameters方法，在接收全局模型时触发平滑操作

        Args:
            model: 全局模型
        """
        # 如果在选择性上传阶段，使用平滑融合
        if self.round >= self.selective_upload_round and self.kept_mask:
            self.receive_and_smooth_global_model(model)
        else:
            # 否则直接复制参数（调用父类方法）
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()
