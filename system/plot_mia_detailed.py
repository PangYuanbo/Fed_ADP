"""
详细的MIA趋势可视化脚本
展示训练过程中的隐私泄露风险变化
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 配置参数
MIA_HISTORY_PATH = "mia_results/cifar-10-shadow_alpha1_rl/mia_history.json"
OUTPUT_DIR = "mia_results/cifar-10-shadow_alpha1_rl"

def load_mia_history(path):
    """加载MIA历史数据"""
    with open(path, 'r') as f:
        data = json.load(f)

    # 过滤掉空的记录
    valid_data = [entry for entry in data if entry.get('summary')]
    return valid_data

def plot_comprehensive_mia_trends(history_data, output_dir):
    """绘制全面的MIA趋势图"""

    # 提取最新一次训练的数据（从round 10开始的连续序列）
    # 找到最后一次从round 10开始的序列
    latest_run_start_idx = None
    for i in range(len(history_data) - 1, -1, -1):
        if history_data[i]['round'] == 10:
            latest_run_start_idx = i
            break

    if latest_run_start_idx is None:
        print("No valid training run found")
        return

    latest_data = history_data[latest_run_start_idx:]

    # 提取数据
    rounds = [entry['round'] for entry in latest_data]
    avg_f_scores = [entry['summary']['avg_f_score'] for entry in latest_data]
    std_f_scores = [entry['summary']['std_f_score'] for entry in latest_data]
    max_f_scores = [entry['summary']['max_f_score'] for entry in latest_data]
    min_f_scores = [entry['summary']['min_f_score'] for entry in latest_data]
    avg_tpr = [entry['summary']['avg_tpr'] for entry in latest_data]
    avg_fpr = [entry['summary']['avg_fpr'] for entry in latest_data]
    avg_accuracy = [entry['summary']['avg_accuracy'] for entry in latest_data]

    # 创建大图
    fig = plt.figure(figsize=(20, 12))

    # ==================== 子图1: F-score趋势（主图） ====================
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rounds, avg_f_scores, 'b-', linewidth=2, label='Average F-score', marker='o')
    ax1.fill_between(rounds,
                      [avg - std for avg, std in zip(avg_f_scores, std_f_scores)],
                      [avg + std for avg, std in zip(avg_f_scores, std_f_scores)],
                      alpha=0.3, label='±1 Std Dev')
    ax1.plot(rounds, max_f_scores, 'r--', linewidth=1, label='Max F-score', alpha=0.7)
    ax1.plot(rounds, min_f_scores, 'g--', linewidth=1, label='Min F-score', alpha=0.7)

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('F-score', fontsize=12)
    ax1.set_title('MIA Attack Success (F-score) vs Training Rounds', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Random Guess (0.5)')

    # 添加最终值注释
    final_f = avg_f_scores[-1]
    ax1.annotate(f'Final: {final_f:.4f}',
                 xy=(rounds[-1], final_f),
                 xytext=(rounds[-1]-20, final_f+0.05),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red', fontweight='bold')

    # ==================== 子图2: TPR vs FPR ====================
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(rounds, avg_tpr, 'r-', linewidth=2, label='True Positive Rate', marker='s')
    ax2.plot(rounds, avg_fpr, 'b-', linewidth=2, label='False Positive Rate', marker='^')
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('TPR vs FPR over Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)

    # ==================== 子图3: Attack Accuracy ====================
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(rounds, avg_accuracy, 'g-', linewidth=2, label='Attack Accuracy', marker='D')
    ax3.set_xlabel('Training Round', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('MIA Attack Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Random Guess')

    # 添加准确率范围着色
    ax3.fill_between(rounds, 0.5, avg_accuracy, where=[acc > 0.5 for acc in avg_accuracy],
                     alpha=0.2, color='red', label='Privacy Risk')

    # ==================== 子图4: 每个客户端的F-score变化 ====================
    ax4 = plt.subplot(2, 3, 4)

    # 提取每个客户端的F-score
    client_ids = list(latest_data[0]['client_f_scores'].keys())
    for client_id in client_ids:
        client_f_scores = [entry['client_f_scores'][client_id] for entry in latest_data]
        ax4.plot(rounds, client_f_scores, linewidth=1.5, label=f'Client {client_id}', marker='o', markersize=4)

    ax4.set_xlabel('Training Round', fontsize=12)
    ax4.set_ylabel('F-score', fontsize=12)
    ax4.set_title('Per-Client MIA F-score Trends', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.5, color='orange', linestyle=':', linewidth=1)

    # ==================== 子图5: 风险等级分布 ====================
    ax5 = plt.subplot(2, 3, 5)

    high_risk = [entry['summary'].get('high_risk_clients', 0) for entry in latest_data]
    medium_risk = [entry['summary'].get('medium_risk_clients', 0) for entry in latest_data]
    low_risk = [entry['summary'].get('low_risk_clients', 0) for entry in latest_data]

    ax5.stackplot(rounds, high_risk, medium_risk, low_risk,
                  labels=['High Risk', 'Medium Risk', 'Low Risk'],
                  colors=['#ff4444', '#ffaa44', '#44ff44'],
                  alpha=0.7)
    ax5.set_xlabel('Training Round', fontsize=12)
    ax5.set_ylabel('Number of Clients', fontsize=12)
    ax5.set_title('Privacy Risk Distribution', fontsize=14, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3, axis='y')

    # ==================== 子图6: 统计摘要 ====================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # 计算统计数据
    initial_f = avg_f_scores[0]
    final_f = avg_f_scores[-1]
    max_f = max(avg_f_scores)
    min_f = min(avg_f_scores)
    avg_f = np.mean(avg_f_scores)

    improvement = final_f - initial_f
    improvement_pct = (improvement / initial_f) * 100

    stats_text = f"""
    ══════════ MIA Evaluation Summary ══════════

    Training Rounds: {rounds[0]} → {rounds[-1]}
    Total Evaluations: {len(rounds)}

    ─────────── F-score Statistics ───────────
    Initial F-score:     {initial_f:.4f}
    Final F-score:       {final_f:.4f}
    Change:              {improvement:+.4f} ({improvement_pct:+.2f}%)

    Maximum F-score:     {max_f:.4f} (Round {rounds[avg_f_scores.index(max_f)]})
    Minimum F-score:     {min_f:.4f} (Round {rounds[avg_f_scores.index(min_f)]})
    Average F-score:     {avg_f:.4f}
    Std Dev:             {np.std(avg_f_scores):.4f}

    ─────────── Final Round Stats ───────────
    TPR:                 {avg_tpr[-1]:.4f}
    FPR:                 {avg_fpr[-1]:.4f}
    Attack Accuracy:     {avg_accuracy[-1]:.4f}

    High Risk Clients:   {high_risk[-1]}
    Medium Risk Clients: {medium_risk[-1]}
    Low Risk Clients:    {low_risk[-1]}

    ═══════════════════════════════════════════

    ⚠️  Privacy Risk Assessment:
    F-score > 0.7: HIGH privacy leakage risk
    F-score 0.5-0.7: MEDIUM privacy leakage risk
    F-score < 0.5: LOW privacy leakage risk
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))

    # 整体标题
    fig.suptitle('Comprehensive Membership Inference Attack (MIA) Analysis\nCIFAR-10 Shadow Models (α=1.0)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    output_path = Path(output_dir) / f"mia_comprehensive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive MIA analysis plot saved to: {output_path}")

    plt.close()

    return output_path

def print_summary_stats(history_data):
    """打印统计摘要"""
    # 找到最新一次训练
    latest_run_start_idx = None
    for i in range(len(history_data) - 1, -1, -1):
        if history_data[i]['round'] == 10:
            latest_run_start_idx = i
            break

    latest_data = history_data[latest_run_start_idx:]

    print("\n" + "="*60)
    print("MIA EVALUATION SUMMARY")
    print("="*60)

    rounds = [entry['round'] for entry in latest_data]
    avg_f_scores = [entry['summary']['avg_f_score'] for entry in latest_data]

    print(f"Total MIA evaluation rounds: {len(rounds)}")
    print(f"Evaluated rounds: {rounds}")
    print(f"Final F-score: {avg_f_scores[-1]:.4f}")
    print(f"Average F-score: {np.mean(avg_f_scores):.4f}")
    print(f"Max F-score: {max(avg_f_scores):.4f}")
    print(f"Min F-score: {min(avg_f_scores):.4f}")
    print(f"F-score std: {np.std(avg_f_scores):.4f}")
    print("="*60)

if __name__ == "__main__":
    print("Loading MIA history data...")
    history = load_mia_history(MIA_HISTORY_PATH)

    print(f"Found {len(history)} MIA evaluation records")

    # 打印统计摘要
    print_summary_stats(history)

    # 绘制详细图表
    print("\nGenerating comprehensive MIA analysis plots...")
    output_path = plot_comprehensive_mia_trends(history, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✓ MIA visualization complete!")
    print(f"✓ Output saved to: {output_path}")
    print("="*60)
