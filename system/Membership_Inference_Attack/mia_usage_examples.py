# ============================
# mia_usage_examples.py
# MIA攻击接口使用示例 - 返回JSON格式结果
# ============================

"""
MIA攻击接口使用示例

这个文件展示了如何在训练过程中使用MIA攻击接口来监控模型的隐私风险。
所有函数都返回JSON格式的结果，不会产生print输出。
"""

from mia_attack_interface import run_mia_attack, run_full_mia_evaluation, quick_mia_test


def example_1_quick_test():
    """
    示例1：快速测试单个模型和标签
    """
    # 快速测试
    result = quick_mia_test('path/to/model.pt', test_label=0)

    # 检查结果
    if result['status'] == 'success':
        f_score = result['f_score']
        tpr = result['tpr']
        fpr = result['fpr']
        # 在这里处理结果，比如记录到日志或发送通知
    else:
        error_msg = result.get('error', 'Unknown error')
        # 处理错误


def example_2_single_attack():
    """
    示例2：单次完整攻击评估
    """
    # 对单个模型进行完整的MIA攻击评估
    result = run_mia_attack(
        target_model_path='path/to/model.pt',
        target_label=7,  # 特定标签，或None表示所有标签
        save_results=True  # 保存结果到文件
    )

    # 获取关键指标
    if result['status'] == 'success':
        summary = result['summary']
        avg_f_score = summary['avg_f_score']
        labels_evaluated = summary['labels_evaluated']
        success_rate = summary['success_rate']

        # 检查是否有错误
        if result['errors']:
            # 部分成功，有一些标签失败了
            for error in result['errors']:
                error_type = error['type']
                error_msg = error['message']

    return result


def example_3_training_loop_integration():
    """
    示例3：在训练循环中集成MIA评估
    """
    # 模拟训练循环
    num_epochs = 100
    mia_check_interval = 10

    for epoch in range(num_epochs):
        # ... 您的训练代码 ...

        # 每10个epoch评估一次MIA风险
        if epoch % mia_check_interval == 0:
            model_path = f"checkpoint_epoch_{epoch}.pt"
            # torch.save(model.state_dict(), model_path)  # 保存模型

            # 执行MIA攻击评估
            mia_result = run_mia_attack(
                target_model_path=model_path,
                target_label=None,  # 评估所有标签
                save_results=True
            )

            # 分析结果
            if mia_result['status'] in ['success', 'partial_success']:
                avg_f_score = mia_result['summary']['avg_f_score']

                # 记录到训练日志
                training_log = {
                    'epoch': epoch,
                    'mia_f_score': avg_f_score,
                    'privacy_risk': 'high' if avg_f_score > 0.8 else 'medium' if avg_f_score > 0.6 else 'low'
                }

                # 如果隐私风险过高，可以采取对策
                if avg_f_score > 0.8:
                    # 实施隐私保护措施
                    take_privacy_protection_measures()
            else:
                # MIA评估失败，记录错误
                error_log = {
                    'epoch': epoch,
                    'mia_status': 'failed',
                    'errors': mia_result.get('errors', [])
                }


def example_4_batch_evaluation():
    """
    示例4：批量评估多个模型
    """
    # 准备多个模型路径
    model_paths = [f"client_{i}_model.pt" for i in range(10)]

    # 执行完整的MIA评估
    results = run_full_mia_evaluation(
        target_model_paths=model_paths,
        retrain_attack=False,  # 使用已有的攻击模型
        save_results=True
    )

    # 分析结果
    if results['status'] in ['success', 'partial_success']:
        # 获取模型准确率
        model_accuracy = results['model_accuracy']
        avg_accuracy = model_accuracy.get('average', 0.0)

        # 获取攻击结果
        summary = results['summary']
        total_evaluations = summary['total_evaluations']
        avg_f_score = summary['avg_f_score']

        # 逐客户端分析
        for client_id, client_results in results['attack_results'].items():
            client_metrics = analyze_client_results(client_results)

        # 整体隐私风险评估
        privacy_assessment = {
            'overall_risk': classify_privacy_risk(avg_f_score),
            'vulnerable_clients': find_vulnerable_clients(results['attack_results']),
            'recommended_actions': get_privacy_recommendations(avg_f_score)
        }

    return results


def example_5_custom_configuration():
    """
    示例5：自定义配置参数
    """
    # 自定义配置
    custom_config = {
        'batch_size': 2,
        'num_classes': 10,
        'alpha': 0.5,  # 不同的数据分布参数
        'attack_epochs': 30,  # 更少的训练轮次用于快速测试
        'attack_lr': 5e-4,
    }

    result = run_mia_attack(
        target_model_path='path/to/model.pt',
        config=custom_config,
        save_results=False  # 不保存文件，仅返回结果
    )

    return result


def example_6_error_handling():
    """
    示例6：错误处理和日志分析
    """
    result = run_mia_attack('path/to/model.pt')

    # 检查执行状态
    if result['status'] == 'success':
        # 完全成功
        process_successful_result(result)
    elif result['status'] == 'partial_success':
        # 部分成功，检查具体错误
        successful_labels = len(result['attack_results'])
        total_errors = len(result['errors'])

        for error in result['errors']:
            handle_specific_error(error)
    else:
        # 完全失败
        handle_complete_failure(result)

    # 分析日志
    for log_entry in result['logs']:
        timestamp = log_entry['timestamp']
        level = log_entry['level']
        message = log_entry['message']
        # 处理日志信息


# ============================
# 辅助函数
# ============================

def take_privacy_protection_measures():
    """实施隐私保护措施"""
    # 例如：增加差分隐私噪声、调整学习率、早停等
    pass


def analyze_client_results(client_results):
    """分析单个客户端的结果"""
    f_scores = [r['f_score'] for r in client_results]
    return {
        'avg_f_score': sum(f_scores) / len(f_scores),
        'max_f_score': max(f_scores),
        'vulnerable_labels': [i for i, score in enumerate(f_scores) if score > 0.8]
    }


def classify_privacy_risk(f_score):
    """分类隐私风险级别"""
    if f_score > 0.8:
        return 'high'
    elif f_score > 0.6:
        return 'medium'
    else:
        return 'low'


def find_vulnerable_clients(attack_results):
    """找出易受攻击的客户端"""
    vulnerable = []
    for client_id, results in attack_results.items():
        client_metrics = analyze_client_results(results)
        if client_metrics['avg_f_score'] > 0.7:
            vulnerable.append(client_id)
    return vulnerable


def get_privacy_recommendations(f_score):
    """根据F-score提供隐私保护建议"""
    if f_score > 0.8:
        return ["使用差分隐私", "减少训练轮次", "增加噪声"]
    elif f_score > 0.6:
        return ["考虑使用差分隐私", "监控模型复杂度"]
    else:
        return ["继续当前设置"]


def process_successful_result(result):
    """处理成功的结果"""
    pass


def handle_specific_error(error):
    """处理特定错误"""
    error_type = error['type']
    if error_type == 'model_load_failed':
        # 处理模型加载失败
        pass
    elif error_type == 'attack_training_failed':
        # 处理攻击模型训练失败
        pass


def handle_complete_failure(result):
    """处理完全失败的情况"""
    pass


# ============================
# 主要使用模式总结
# ============================

def main_usage_patterns():
    """
    主要使用模式总结：

    1. 快速测试：quick_mia_test() - 验证单个标签
    2. 单次评估：run_mia_attack() - 完整评估单个模型
    3. 批量评估：run_full_mia_evaluation() - 评估多个模型
    4. 训练集成：在训练循环中定期调用
    5. 自定义配置：传入config参数调整行为
    6. 错误处理：检查status和errors字段

    所有函数都返回JSON格式的结构化结果，便于程序化处理。
    """
    return {
        'interface_type': 'json_only',
        'output_format': 'structured_dict',
        'error_handling': 'comprehensive',
        'logging': 'built_in',
        'customization': 'flexible'
    }


if __name__ == "__main__":
    # 本文件仅作为示例参考，不执行任何代码
    pass