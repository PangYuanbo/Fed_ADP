# This is a patch file for evaluate_client_mia method
# Line 169-216 replacement

    def evaluate_client_mia(self,
                           client,
                           target_labels: Optional[List[int]] = None,
                           global_test_loader = None) -> Dict:
        """
        评估单个客户端的MIA攻击成功率

        Args:
            client: 客户端对象，必须有以下属性/方法：
                - client.id: 客户端ID
                - client.model: 客户端模型
                - client.load_train_data(): 返回训练数据DataLoader
                - client.load_test_data(): 返回测试数据DataLoader
            target_labels: 目标标签列表，None表示所有标签
            global_test_loader: 全局test数据加载器（如果提供且启用use_global_test，将使用它代替client自己的test）

        Returns:
            Dict: MIA评估结果
        """
        if not self.attack_models:
            print(f"[MIA Wrapper] Client {client.id}: No attack models available")
            return {
                'client_id': client.id,
                'status': 'failed',
                'error': 'No attack models available'
            }

        # 确定要评估的标签
        if target_labels is None:
            target_labels = list(self.attack_models.keys())
        else:
            target_labels = [label for label in target_labels if label in self.attack_models]

        if not target_labels:
            print(f"[MIA Wrapper] Client {client.id}: No valid target labels")
            return {
                'client_id': client.id,
                'status': 'failed',
                'error': 'No valid target labels'
            }

        try:
            # 🔑 关键修复: 评估前清理模型梯度和缓存
            if hasattr(client, 'model') and client.model is not None:
                client.model.eval()  # 确保评估模式
                client.model.zero_grad()  # 清理梯度
                # 清理所有参数的梯度
                for param in client.model.parameters():
                    if param.grad is not None:
                        param.grad = None

            # 获取client的train数据加载器
            train_loader = client.load_train_data(batch_size=self.batch_size)

            # 根据配置选择test数据源
            if self.use_global_test and global_test_loader is not None:
                test_loader = global_test_loader
                print(f"[MIA Wrapper] Client {client.id}: Using GLOBAL test data (all clients combined)")
            else:
                test_loader = client.load_test_data(batch_size=self.batch_size)
                print(f"[MIA Wrapper] Client {client.id}: Using client's OWN test data")

            print(f"[MIA Wrapper] Client {client.id}: Data loaders created successfully")
