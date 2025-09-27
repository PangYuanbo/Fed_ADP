# ============================
# mia_visualizer.py
# MIA结果可视化工具 - 绘制F-score变化趋势图
# ============================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class MIAVisualizer:
    """
    MIA结果可视化器
    """

    def __init__(self):
        """初始化可视化器"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')

        # 设置颜色主题
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#9B59B6',
            'warning': '#F39C12',
            'danger': '#E74C3C'
        }

        # 风险级别颜色
        self.risk_colors = {
            'high': '#E74C3C',    # 红色
            'medium': '#F39C12',  # 橙色
            'low': '#27AE60'      # 绿色
        }

    def plot_f_score_trends(self,
                           mia_results_history: List[Dict],
                           save_dir: str = "mia_visualizations",
                           dataset_name: str = "dataset",
                           alpha: float = 1.0) -> str:
        """
        绘制F-score变化趋势图

        Args:
            mia_results_history: MIA结果历史数据
            save_dir: 保存目录
            dataset_name: 数据集名称
            alpha: 数据分布参数

        Returns:
            str: 保存的图片路径
        """
        if not mia_results_history:
            raise ValueError("No MIA results history provided")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 提取数据
        rounds = []
        avg_f_scores = []
        client_f_scores = {}  # client_id -> [f_scores]
        risk_distributions = []  # [(high, medium, low), ...]

        for result in mia_results_history:
            if result['status'] == 'success' and 'summary' in result:
                round_num = result['round']
                summary = result['summary']

                rounds.append(round_num)
                avg_f_scores.append(summary['avg_f_score'])

                # 收集风险分布数据
                risk_dist = (
                    summary.get('high_risk_clients', 0),
                    summary.get('medium_risk_clients', 0),
                    summary.get('low_risk_clients', 0)
                )
                risk_distributions.append(risk_dist)

                # 收集各客户端数据
                for client_id, client_result in result['clients'].items():
                    if client_result['status'] == 'success':
                        if client_id not in client_f_scores:
                            client_f_scores[client_id] = []
                        client_f_scores[client_id].append(
                            client_result['summary']['avg_f_score']
                        )

        if not rounds:
            raise ValueError("No valid MIA results found")

        # 创建主图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 平均F-score趋势图
        self._plot_average_f_score_trend(ax1, rounds, avg_f_scores)

        # 2. 风险分布堆叠图
        self._plot_risk_distribution(ax2, rounds, risk_distributions)

        # 3. 各客户端F-score趋势
        self._plot_individual_client_trends(ax3, rounds, client_f_scores)

        # 4. F-score分布箱线图
        self._plot_f_score_distribution(ax4, rounds, client_f_scores)

        # 设置总标题
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f'MIA Attack F-score Analysis - {dataset_name} (α={alpha})\n'
                    f'Generated at {timestamp}', fontsize=16, fontweight='bold')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mia_trends_{dataset_name}_alpha{alpha}_{timestamp_str}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        return save_path

    def _plot_average_f_score_trend(self, ax, rounds, avg_f_scores):
        """绘制平均F-score趋势"""
        ax.plot(rounds, avg_f_scores,
               marker='o', linewidth=2.5, markersize=6,
               color=self.colors['primary'], markerfacecolor='white',
               markeredgewidth=2, markeredgecolor=self.colors['primary'])

        # 添加风险级别的水平线
        ax.axhline(y=0.8, color=self.risk_colors['high'], linestyle='--', alpha=0.7, label='High Risk (>0.8)')
        ax.axhline(y=0.6, color=self.risk_colors['medium'], linestyle='--', alpha=0.7, label='Medium Risk (>0.6)')

        # 填充风险区域
        ax.fill_between(rounds, 0.8, 1.0, alpha=0.1, color=self.risk_colors['high'])
        ax.fill_between(rounds, 0.6, 0.8, alpha=0.1, color=self.risk_colors['medium'])
        ax.fill_between(rounds, 0.0, 0.6, alpha=0.1, color=self.risk_colors['low'])

        ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average F-score', fontsize=12, fontweight='bold')
        ax.set_title('Average MIA F-score Trend Across All Clients', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)

        # 添加数值标注
        for i, (round_num, f_score) in enumerate(zip(rounds, avg_f_scores)):
            if i % 2 == 0 or i == len(rounds) - 1:
                ax.annotate(f'{f_score:.3f}',
                          (round_num, f_score),
                          textcoords="offset points",
                          xytext=(0,10),
                          ha='center',
                          fontsize=9,
                          alpha=0.8)

    def _plot_risk_distribution(self, ax, rounds, risk_distributions):
        """绘制风险分布堆叠图"""
        if not risk_distributions:
            ax.text(0.5, 0.5, 'No Risk Data Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7)
            return

        high_counts = [dist[0] for dist in risk_distributions]
        medium_counts = [dist[1] for dist in risk_distributions]
        low_counts = [dist[2] for dist in risk_distributions]

        ax.stackplot(rounds,
                    high_counts, medium_counts, low_counts,
                    labels=['High Risk', 'Medium Risk', 'Low Risk'],
                    colors=[self.risk_colors['high'], self.risk_colors['medium'], self.risk_colors['low']],
                    alpha=0.8)

        ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Clients', fontsize=12, fontweight='bold')
        ax.set_title('Client Risk Distribution Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_individual_client_trends(self, ax, rounds, client_f_scores):
        """绘制各客户端F-score趋势"""
        if not client_f_scores:
            ax.text(0.5, 0.5, 'No Client Data Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7)
            return

        # 选择要显示的客户端（最多显示10个）
        client_ids = sorted(list(client_f_scores.keys()))
        if len(client_ids) > 10:
            step = len(client_ids) // 10
            selected_clients = client_ids[::step][:10]
        else:
            selected_clients = client_ids

        # 生成颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_clients)))

        for i, client_id in enumerate(selected_clients):
            f_scores = client_f_scores[client_id]
            client_rounds = rounds[:len(f_scores)]
            ax.plot(client_rounds, f_scores,
                   marker='o', linewidth=1.5, markersize=4,
                   color=colors[i], alpha=0.8, label=f'Client {client_id}')

        ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('F-score', fontsize=12, fontweight='bold')
        ax.set_title('Individual Client F-score Trends', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)

    def _plot_f_score_distribution(self, ax, rounds, client_f_scores):
        """绘制F-score分布箱线图"""
        if not client_f_scores:
            ax.text(0.5, 0.5, 'No Client Data Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7)
            return

        # 为每轮准备数据
        round_data = []
        round_labels = []

        for i, round_num in enumerate(rounds):
            round_f_scores = []
            for client_scores in client_f_scores.values():
                if i < len(client_scores):
                    round_f_scores.append(client_scores[i])

            if round_f_scores:
                round_data.append(round_f_scores)
                round_labels.append(f'R{round_num}')

        if round_data:
            box_plot = ax.boxplot(round_data, labels=round_labels, patch_artist=True)

            # 设置箱线图颜色
            for patch in box_plot['boxes']:
                patch.set_facecolor(self.colors['primary'])
                patch.set_alpha(0.7)

            ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
            ax.set_ylabel('F-score Distribution', fontsize=12, fontweight='bold')
            ax.set_title('F-score Distribution Across Clients by Round', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            # 旋转x轴标签以避免重叠
            plt.setp(ax.get_xticklabels(), rotation=45)

    def create_summary_report(self,
                            mia_results_history: List[Dict],
                            save_dir: str = "mia_visualizations",
                            dataset_name: str = "dataset",
                            alpha: float = 1.0) -> str:
        """
        创建MIA评估总结报告

        Args:
            mia_results_history: MIA结果历史数据
            save_dir: 保存目录
            dataset_name: 数据集名称
            alpha: 数据分布参数

        Returns:
            str: 保存的报告路径
        """
        if not mia_results_history:
            raise ValueError("No MIA results history provided")

        os.makedirs(save_dir, exist_ok=True)

        # 生成报告内容
        report = []
        report.append(f"# MIA Attack Evaluation Report")
        report.append(f"Dataset: {dataset_name} (α={alpha})")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 总体统计
        all_f_scores = []
        for result in mia_results_history:
            if result['status'] == 'success' and 'summary' in result:
                all_f_scores.append(result['summary']['avg_f_score'])

        if all_f_scores:
            report.append("## Overall Statistics")
            report.append(f"- Total evaluation rounds: {len(all_f_scores)}")
            report.append(f"- Average F-score: {np.mean(all_f_scores):.4f}")
            report.append(f"- Maximum F-score: {np.max(all_f_scores):.4f}")
            report.append(f"- Minimum F-score: {np.min(all_f_scores):.4f}")
            report.append(f"- Standard deviation: {np.std(all_f_scores):.4f}")
            report.append("")

        # 逐轮详细结果
        report.append("## Round-by-Round Results")
        report.append("| Round | Avg F-score | High Risk | Medium Risk | Low Risk | Total Clients |")
        report.append("|-------|-------------|-----------|-------------|----------|---------------|")

        for result in mia_results_history:
            if result['status'] == 'success' and 'summary' in result:
                summary = result['summary']
                report.append(f"| {result['round']} | {summary['avg_f_score']:.4f} | "
                            f"{summary.get('high_risk_clients', 0)} | "
                            f"{summary.get('medium_risk_clients', 0)} | "
                            f"{summary.get('low_risk_clients', 0)} | "
                            f"{summary['total_clients']} |")

        # 保存报告
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mia_report_{dataset_name}_alpha{alpha}_{timestamp_str}.md"
        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        return save_path