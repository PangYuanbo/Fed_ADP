"""
绘制200轮RL训练中MIA攻击准确率的变化趋势
针对 cifar-10-shadow_alpha1_rl 数据集
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

def load_and_filter_mia_history(path):
    """加载MIA历史数据并过滤有效记录"""
    with open(path, 'r') as f:
        data = json.load(f)

    # 只保留有summary的有效记录
    valid_data = [entry for entry in data if entry.get('summary') and entry['summary']]

    # 找到最近一次从round 10开始到round 200的连续序列
    latest_run = []
    for i in range(len(valid_data) - 1, -1, -1):
        if valid_data[i]['round'] == 200:
            # 向前收集完整序列
            j = i
            while j >= 0 and valid_data[j]['round'] >= 10:
                latest_run.insert(0, valid_data[j])
                j -= 1
            break

    # 如果没找到完整序列,返回所有有效数据
    if not latest_run:
        latest_run = valid_data

    # 按round排序并去重(保留每个round最新的记录)
    round_dict = {}
    for entry in latest_run:
        round_num = entry['round']
        if round_num not in round_dict or entry['timestamp'] > round_dict[round_num]['timestamp']:
            round_dict[round_num] = entry

    # 转换回列表并排序
    unique_data = sorted(round_dict.values(), key=lambda x: x['round'])

    return unique_data

def plot_comprehensive_mia_trends(history_data, output_dir):
    """绘制全面的MIA趋势图"""

    if not history_data:
        print("❌ No valid data found")
        return None

    # 提取数据
    rounds = [entry['round'] for entry in history_data]
    avg_f_scores = [entry['summary']['avg_f_score'] for entry in history_data]
    std_f_scores = [entry['summary']['std_f_score'] for entry in history_data]
    max_f_scores = [entry['summary']['max_f_score'] for entry in history_data]
    min_f_scores = [entry['summary']['min_f_score'] for entry in history_data]
    avg_tpr = [entry['summary']['avg_tpr'] for entry in history_data]
    avg_fpr = [entry['summary']['avg_fpr'] for entry in history_data]
    avg_accuracy = [entry['summary']['avg_accuracy'] for entry in history_data]

    # 风险等级分布
    high_risk = [entry['summary'].get('high_risk_clients', 0) for entry in history_data]
    medium_risk = [entry['summary'].get('medium_risk_clients', 0) for entry in history_data]
    low_risk = [entry['summary'].get('low_risk_clients', 0) for entry in history_data]

    # 创建大图
    fig = plt.figure(figsize=(22, 14))

    # ==================== 子图1: F-score趋势（主图） ====================
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rounds, avg_f_scores, 'b-', linewidth=2.5, label='Average F-score', marker='o', markersize=5)
    ax1.fill_between(rounds,
                      [avg - std for avg, std in zip(avg_f_scores, std_f_scores)],
                      [avg + std for avg, std in zip(avg_f_scores, std_f_scores)],
                      alpha=0.3, label='±1 Std Dev', color='blue')
    ax1.plot(rounds, max_f_scores, 'r--', linewidth=1.5, label='Max F-score', alpha=0.7)
    ax1.plot(rounds, min_f_scores, 'g--', linewidth=1.5, label='Min F-score', alpha=0.7)

    # 添加风险级别参考线
    ax1.axhline(y=0.8, color='red', linestyle=':', linewidth=2, alpha=0.6, label='High Risk (>0.8)')
    ax1.axhline(y=0.6, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Medium Risk (>0.6)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Random Guess (0.5)')

    ax1.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('F-score', fontsize=13, fontweight='bold')
    ax1.set_title('MIA Attack Success (F-score) vs Training Rounds', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 添加最终值注释
    final_f = avg_f_scores[-1]
    initial_f = avg_f_scores[0]
    ax1.annotate(f'Final: {final_f:.4f}',
                 xy=(rounds[-1], final_f),
                 xytext=(rounds[-1]-30, final_f-0.1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=11, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax1.annotate(f'Initial: {initial_f:.4f}',
                 xy=(rounds[0], initial_f),
                 xytext=(rounds[0]+20, initial_f+0.1),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                 fontsize=11, color='blue', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    # ==================== 子图2: TPR vs FPR ====================
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(rounds, avg_tpr, 'r-', linewidth=2.5, label='True Positive Rate (TPR)', marker='s', markersize=4)
    ax2.plot(rounds, avg_fpr, 'b-', linewidth=2.5, label='False Positive Rate (FPR)', marker='^', markersize=4)
    ax2.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Rate', fontsize=13, fontweight='bold')
    ax2.set_title('TPR vs FPR over Training', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_ylim(0, 1.1)

    # ==================== 子图3: Attack Accuracy ====================
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(rounds, avg_accuracy, 'g-', linewidth=2.5, label='Attack Accuracy', marker='D', markersize=4)
    ax3.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax3.set_title('MIA Attack Accuracy', fontsize=15, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Random Guess')

    # 添加准确率范围着色
    ax3.fill_between(rounds, 0.5, avg_accuracy, where=[acc > 0.5 for acc in avg_accuracy],
                     alpha=0.25, color='red', label='Privacy Risk')
    ax3.set_ylim(0, 1)

    # ==================== 子图4: 每个客户端的F-score变化 ====================
    ax4 = plt.subplot(2, 3, 4)

    # 提取每个客户端的F-score
    if history_data[0].get('client_f_scores'):
        client_ids = list(history_data[0]['client_f_scores'].keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(client_ids)))

        for idx, client_id in enumerate(client_ids):
            client_f_scores = []
            for entry in history_data:
                if 'client_f_scores' in entry and client_id in entry['client_f_scores']:
                    client_f_scores.append(entry['client_f_scores'][client_id])
                else:
                    client_f_scores.append(None)

            # 过滤None值
            valid_rounds = [r for r, f in zip(rounds, client_f_scores) if f is not None]
            valid_scores = [f for f in client_f_scores if f is not None]

            ax4.plot(valid_rounds, valid_scores, linewidth=2, label=f'Client {client_id}',
                    marker='o', markersize=3, color=colors[idx])

    ax4.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax4.set_ylabel('F-score', fontsize=13, fontweight='bold')
    ax4.set_title('Per-Client MIA F-score Trends', fontsize=15, fontweight='bold')
    ax4.legend(loc='best', ncol=1, fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5)
    ax4.axhline(y=0.6, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_ylim(0, 1)

    # ==================== 子图5: 风险等级分布 ====================
    ax5 = plt.subplot(2, 3, 5)

    ax5.stackplot(rounds, high_risk, medium_risk, low_risk,
                  labels=['High Risk (F>0.8)', 'Medium Risk (0.6<F≤0.8)', 'Low Risk (F≤0.6)'],
                  colors=['#ff4444', '#ffaa44', '#44ff44'],
                  alpha=0.8)
    ax5.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Number of Clients', fontsize=13, fontweight='bold')
    ax5.set_title('Privacy Risk Distribution', fontsize=15, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=10)
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
    improvement_pct = (improvement / initial_f) * 100 if initial_f != 0 else 0

    # 找到最大值的轮次
    max_round = rounds[avg_f_scores.index(max_f)]
    min_round = rounds[avg_f_scores.index(min_f)]

    stats_text = f"""
    ══════════ MIA Evaluation Summary ══════════

    Training Rounds: {rounds[0]} → {rounds[-1]}
    Total Evaluations: {len(rounds)}

    ─────────── F-score Statistics ───────────
    Initial F-score:     {initial_f:.4f}
    Final F-score:       {final_f:.4f}
    Change:              {improvement:+.4f} ({improvement_pct:+.2f}%)

    Maximum F-score:     {max_f:.4f} (Round {max_round})
    Minimum F-score:     {min_f:.4f} (Round {min_round})
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
    F-score > 0.8: HIGH privacy leakage risk
    F-score 0.6-0.8: MEDIUM privacy leakage risk
    F-score < 0.6: LOW privacy leakage risk

    📊 Trend Analysis:
    {"✓ Privacy IMPROVED" if improvement < 0 else "⚠ Privacy WORSENED"} over training
    """

    ax6.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.4, edgecolor='brown', linewidth=2))

    # 整体标题
    fig.suptitle('Comprehensive Membership Inference Attack (MIA) Analysis\n'
                 'CIFAR-10 with RL-based Differential Privacy (200 Training Rounds)',
                 fontsize=17, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    output_path = Path(output_dir) / "mia_200rounds_comprehensive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Comprehensive MIA analysis plot saved to: {output_path}")

    plt.close()

    return output_path

def print_summary_stats(history_data):
    """打印统计摘要"""

    print("\n" + "="*80)
    print("MIA EVALUATION SUMMARY - 200 ROUNDS RL TRAINING")
    print("="*80)

    rounds = [entry['round'] for entry in history_data]
    avg_f_scores = [entry['summary']['avg_f_score'] for entry in history_data]

    print(f"Dataset: CIFAR-10 with Shadow Models (α=1.0)")
    print(f"Total MIA evaluation rounds: {len(rounds)}")
    print(f"Round range: {min(rounds)} - {max(rounds)}")
    print(f"\nF-score Evolution:")
    print(f"  Initial F-score (Round {rounds[0]}): {avg_f_scores[0]:.4f}")
    print(f"  Final F-score (Round {rounds[-1]}):   {avg_f_scores[-1]:.4f}")
    print(f"  Change: {avg_f_scores[-1] - avg_f_scores[0]:+.4f} ({((avg_f_scores[-1] - avg_f_scores[0]) / avg_f_scores[0] * 100):+.2f}%)")
    print(f"\nOverall Statistics:")
    print(f"  Average F-score: {np.mean(avg_f_scores):.4f}")
    print(f"  Max F-score: {max(avg_f_scores):.4f} (Round {rounds[avg_f_scores.index(max(avg_f_scores))]})")
    print(f"  Min F-score: {min(avg_f_scores):.4f} (Round {rounds[avg_f_scores.index(min(avg_f_scores))]})")
    print(f"  F-score std: {np.std(avg_f_scores):.4f}")

    # 风险评估
    final_f = avg_f_scores[-1]
    if final_f > 0.8:
        risk_level = "🔴 HIGH RISK"
    elif final_f > 0.6:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"

    print(f"\nFinal Privacy Risk Level: {risk_level}")
    print("="*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MIA 200-Rounds Analysis Visualization Tool")
    print("="*80)

    print("\n📂 Loading MIA history data...")
    history = load_and_filter_mia_history(MIA_HISTORY_PATH)

    if not history:
        print("❌ No valid MIA evaluation records found!")
        exit(1)

    print(f"✓ Found {len(history)} valid MIA evaluation records")

    # 打印统计摘要
    print_summary_stats(history)

    # 绘制详细图表
    print("\n📊 Generating comprehensive MIA analysis plots...")
    output_path = plot_comprehensive_mia_trends(history, OUTPUT_DIR)

    if output_path:
        print("\n" + "="*80)
        print("✅ MIA visualization complete!")
        print(f"📈 Output saved to: {output_path}")
        print("="*80 + "\n")
    else:
        print("\n❌ Failed to generate visualization")
