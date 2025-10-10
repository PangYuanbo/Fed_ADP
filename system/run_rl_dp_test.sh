#!/bin/bash
# ============================
# run_rl_dp_test.sh
# RL-DP系统快速测试脚本
# ============================

echo "============================================"
echo "RL-DP System Quick Test"
echo "============================================"

# 测试RL-DP系统基本功能
echo "1. Testing RL-DP system components..."
python test_rl_dp_system.py --test all

if [ $? -eq 0 ]; then
    echo "✅ RL-DP system tests passed!"
else
    echo "❌ RL-DP system tests failed!"
    exit 1
fi

echo ""
echo "2. Running quick RL-DP training test..."

# 运行一个快速的训练测试 (5轮，少量客户端)
python main_rl.py \
    --dataset cifar10 \
    --model cnn \
    --num_clients 5 \
    --global_rounds 20 \
    --local_learning_rate 0.01 \
    --difference_privacy True \
    --enable_rl_dp True \
    --enable_mia True \
    --rl_min_rounds 5 \
    --rl_update_interval 3 \
    --preset balanced \
    --device cpu \
    --eval_gap 5

echo ""
echo "============================================"
echo "RL-DP System Test Completed!"
echo "============================================"

# 检查生成的文件
echo "Generated files:"
echo "- RL checkpoints: $(ls -la rl_checkpoints/ 2>/dev/null | wc -l) files"
echo "- MIA results: $(ls -la mia_results/ 2>/dev/null | wc -l) files"
echo "- RL summaries: $(ls -la rl_summaries/ 2>/dev/null | wc -l) files"
echo "- Model checkpoints: $(ls -la pretrain/ 2>/dev/null | wc -l) directories"

echo ""
echo "Next steps:"
echo "1. For full training: python main_rl.py --preset balanced --global_rounds 100"
echo "2. For privacy-focused training: python main_rl.py --preset privacy_focused"
echo "3. For accuracy-focused training: python main_rl.py --preset accuracy_focused"
echo "4. For custom config: edit utils/rl_config.py"