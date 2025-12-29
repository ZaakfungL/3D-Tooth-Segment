#!/usr/bin/env python
"""
从实验日志中提取结果并计算统计数据（均值±标准差）
用法: python scripts/compute_statistics.py results/unet3D_train
"""

import sys
import os
import re
import glob
import numpy as np
from pathlib import Path

def extract_dice_from_log(log_file):
    """从日志文件中提取最佳Dice分数"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配 "最佳模型 Dice: 0.9876 于 Iteration 1234"
        match = re.search(r'最佳模型 Dice:\s*([\d.]+)', content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"警告: 无法从 {log_file} 提取数据: {e}")
    return None

def extract_seed_from_filename(filename):
    """从文件名中提取seed"""
    match = re.search(r'seed(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/compute_statistics.py <日志目录>")
        print("示例: python scripts/compute_statistics.py results/unet3D_train")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    if not os.path.exists(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        sys.exit(1)
    
    # 查找所有日志文件
    log_files = glob.glob(os.path.join(log_dir, "*seed*.log"))
    
    if not log_files:
        print(f"错误: 在 {log_dir} 中未找到日志文件")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"实验结果统计分析")
    print(f"目录: {log_dir}")
    print(f"{'='*60}\n")
    
    # 收集每个seed的最佳结果
    results = {}
    
    for log_file in sorted(log_files):
        seed = extract_seed_from_filename(os.path.basename(log_file))
        dice = extract_dice_from_log(log_file)
        
        if seed is not None and dice is not None:
            # 如果同一个seed有多个日志，保留最新的
            if seed not in results or results[seed]['file'] < log_file:
                results[seed] = {
                    'dice': dice,
                    'file': os.path.basename(log_file)
                }
    
    if not results:
        print("错误: 未能从日志中提取任何有效数据")
        sys.exit(1)
    
    # 按seed排序并显示
    print("各次实验结果:")
    print("-" * 60)
    
    dice_values = []
    for seed in sorted(results.keys()):
        dice = results[seed]['dice']
        dice_values.append(dice)
        print(f"  Seed {seed:4d}: Dice = {dice:.4f}  [{results[seed]['file']}]")
    
    # 计算统计数据
    dice_array = np.array(dice_values)
    mean_dice = np.mean(dice_array)
    std_dice = np.std(dice_array, ddof=1)  # 使用样本标准差 (n-1)
    min_dice = np.min(dice_array)
    max_dice = np.max(dice_array)
    
    print("\n" + "="*60)
    print("统计结果:")
    print("-" * 60)
    print(f"  实验次数:  {len(dice_values)}")
    print(f"  平均值:    {mean_dice:.4f}")
    print(f"  标准差:    {std_dice:.4f}")
    print(f"  最小值:    {min_dice:.4f}")
    print(f"  最大值:    {max_dice:.4f}")
    print("="*60)
    
    # 论文格式输出
    print(f"\n📊 论文中的表示:")
    print(f"   Dice: {mean_dice*100:.2f} ± {std_dice*100:.2f}%")
    print(f"   或写作: {mean_dice:.4f} ± {std_dice:.4f}")
    
    # 保存结果
    output_file = os.path.join(log_dir, "statistics.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("实验结果统计分析\n")
        f.write("="*60 + "\n\n")
        
        f.write("各次实验结果:\n")
        for seed in sorted(results.keys()):
            dice = results[seed]['dice']
            f.write(f"  Seed {seed}: Dice = {dice:.4f}\n")
        
        f.write(f"\n统计结果:\n")
        f.write(f"  实验次数: {len(dice_values)}\n")
        f.write(f"  平均值:   {mean_dice:.4f}\n")
        f.write(f"  标准差:   {std_dice:.4f}\n")
        f.write(f"  最小值:   {min_dice:.4f}\n")
        f.write(f"  最大值:   {max_dice:.4f}\n")
        f.write(f"\n论文格式:\n")
        f.write(f"  Dice: {mean_dice*100:.2f} ± {std_dice*100:.2f}%\n")
    
    print(f"\n✓ 统计结果已保存至: {output_file}\n")

if __name__ == "__main__":
    main()
