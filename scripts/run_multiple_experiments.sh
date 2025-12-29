#!/bin/bash

# =============================================================================
# 通用多次实验自动运行脚本
# 功能: 使用不同随机种子运行多次实验，自动跳过已完成的实验
# 用法: ./run_multiple_experiments.sh [脚本路径] [seed1 seed2 seed3 ...]
# 示例: ./run_multiple_experiments.sh scripts/unet3D_train.py 2025 2026 2027
# =============================================================================

# 1. 解析命令行参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供脚本路径"
    echo "用法: $0 <脚本路径> [seed1 seed2 ...]"
    echo "示例: $0 scripts/unet3D_train.py 2025 2026 2027 2028 2029"
    exit 1
fi

SCRIPT_PATH="$1"
shift  # 移除第一个参数，剩下的都是seeds

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 脚本文件不存在: $SCRIPT_PATH"
    exit 1
fi

# 如果没有提供seeds，使用默认值
if [ $# -eq 0 ]; then
    SEEDS=(2025 2026 2027 2028 2029)
    echo "未指定seed，使用默认值: ${SEEDS[@]}"
else
    SEEDS=("$@")
    echo "使用指定的seeds: ${SEEDS[@]}"
fi

# 2. 根据脚本名自动生成日志目录
SCRIPT_BASENAME=$(basename "$SCRIPT_PATH" .py)
LOG_DIR="results/${SCRIPT_BASENAME}"
mkdir -p "$LOG_DIR"

echo ""
echo "=========================================="
echo "脚本路径: $SCRIPT_PATH"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 3. 检查已完成的实验，自动跳过
echo ""
echo "正在检查已完成的实验..."
PENDING_SEEDS=()
SKIPPED_COUNT=0

for seed in "${SEEDS[@]}"
do
    # 检查是否存在该seed的成功日志
    LOG_PATTERN="${LOG_DIR}/*seed${seed}*.log"
    MATCHING_LOGS=($(ls $LOG_PATTERN 2>/dev/null))
    
    SKIP_THIS_SEED=false
    if [ ${#MATCHING_LOGS[@]} -gt 0 ]; then
        # 检查日志中是否包含"训练结束"标记
        for log in "${MATCHING_LOGS[@]}"; do
            if grep -q "训练结束" "$log" 2>/dev/null; then
                echo "  ✓ Seed $seed 已完成，跳过 (日志: $(basename $log))"
                SKIP_THIS_SEED=true
                SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                break
            fi
        done
    fi
    
    if [ "$SKIP_THIS_SEED" = false ]; then
        PENDING_SEEDS+=($seed)
    fi
done

echo ""
if [ ${#PENDING_SEEDS[@]} -eq 0 ]; then
    echo "所有实验已完成，无需运行！"
    exit 0
fi

echo "待运行: ${#PENDING_SEEDS[@]} 个实验 (已跳过: $SKIPPED_COUNT)"
echo "Seeds: ${PENDING_SEEDS[@]}"

# 记录开始时间
START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "开始执行 ${#PENDING_SEEDS[@]} 次训练任务"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 4. 循环运行待执行的实验
for seed in "${PENDING_SEEDS[@]}"
do
    # 生成带时间戳的日志文件名，避免覆盖
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    CURRENT_LOG="${LOG_DIR}/${SCRIPT_BASENAME}_seed${seed}_${TIMESTAMP}.log"
    
    echo ""
    echo "--------------------------------------"
    echo "正在运行 Seed: ${seed}"
    echo "日志保存至: $CURRENT_LOG"
    echo "开始时间: $(date '+%H:%M:%S')"
    echo "--------------------------------------"
    
    # 执行 Python 脚本，传入 seed 参数
    python -u "$SCRIPT_PATH" --seed "$seed" > "$CURRENT_LOG" 2>&1
    
    # 检查上一步是否成功执行
    if [ $? -eq 0 ]; then
        echo "✓ Seed ${seed} 运行成功完成 ($(date '+%H:%M:%S'))"
        
        # 提取最佳 Dice 分数（如果日志中包含）
        if grep -q "最佳模型 Dice" "$CURRENT_LOG"; then
            best_dice=$(grep "最佳模型 Dice" "$CURRENT_LOG" | tail -1)
            echo "  结果: $best_dice"
        fi
    else
        echo "✗ Seed ${seed} 运行出现错误，请检查日志: $CURRENT_LOG"
    fi
done

# 计算总用时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=========================================="
echo "所有 ${#PENDING_SEEDS[@]} 次训练任务已完成！"
echo "总用时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 5. 汇总所有结果
echo ""
echo "正在汇总实验结果..."
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"

echo "=========================================" > "$SUMMARY_FILE"
echo "实验结果汇总 - $(basename $SCRIPT_PATH)" >> "$SUMMARY_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 提取所有seeds的最佳结果（包括之前完成的）
for seed in "${SEEDS[@]}"
do
    LOG_PATTERN="${LOG_DIR}/*seed${seed}*.log"
    MATCHING_LOGS=($(ls $LOG_PATTERN 2>/dev/null))
    
    if [ ${#MATCHING_LOGS[@]} -gt 0 ]; then
        # 使用最新的日志文件
        LATEST_LOG=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)
        
        echo "Seed ${seed}:" >> "$SUMMARY_FILE"
        echo "  日志: $(basename $LATEST_LOG)" >> "$SUMMARY_FILE"
        
        # 提取最佳 Dice 和对应的 iteration
        if grep -q "最佳模型 Dice" "$LATEST_LOG" 2>/dev/null; then
            grep "最佳模型 Dice" "$LATEST_LOG" | tail -1 >> "$SUMMARY_FILE"
        else
            echo "  未找到结果数据" >> "$SUMMARY_FILE"
        fi
        
        # 提取总训练时间
        if grep -q "训练结束。总用时" "$LATEST_LOG" 2>/dev/null; then
            grep "训练结束。总用时" "$LATEST_LOG" | tail -1 >> "$SUMMARY_FILE"
        fi
        
        echo "" >> "$SUMMARY_FILE"
    else
        echo "Seed ${seed}: 未运行" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "结果汇总已保存至: $SUMMARY_FILE"
echo ""
echo "查看汇总: cat $SUMMARY_FILE"
