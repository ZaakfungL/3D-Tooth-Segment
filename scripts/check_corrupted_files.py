#!/usr/bin/env python3
"""检查数据集中损坏的 .nii.gz 文件"""
import os
import glob
import nibabel as nib
from tqdm import tqdm

DATA_DIR = "/home/ta/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"

def check_file(filepath):
    """检查单个文件是否损坏"""
    try:
        img = nib.load(filepath)
        # 尝试读取数据
        _ = img.get_fdata()
        return True, None
    except Exception as e:
        return False, str(e)

def check_directory(dir_path, label=""):
    """检查目录中的所有文件"""
    if not os.path.exists(dir_path):
        print(f"⚠️ 目录不存在: {dir_path}")
        return
    
    files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
    print(f"\n{'='*60}")
    print(f"检查 {label}: {dir_path}")
    print(f"文件总数: {len(files)}")
    print(f"{'='*60}\n")
    
    corrupted_files = []
    
    for filepath in tqdm(files, desc=f"检查 {label}"):
        is_ok, error = check_file(filepath)
        if not is_ok:
            corrupted_files.append((filepath, error))
            print(f"\n❌ 损坏: {os.path.basename(filepath)}")
            print(f"   错误: {error[:100]}")
    
    if corrupted_files:
        print(f"\n{'='*60}")
        print(f"⚠️ 发现 {len(corrupted_files)} 个损坏文件:")
        print(f"{'='*60}")
        for fpath, err in corrupted_files:
            print(f"  - {fpath}")
    else:
        print(f"\n✅ 所有文件完好!")
    
    return corrupted_files

if __name__ == "__main__":
    print("🔍 开始检查数据集完整性...")
    
    all_corrupted = []
    
    # 检查训练图像
    corrupted = check_directory(
        os.path.join(DATA_DIR, "imagesTr"),
        "训练图像"
    )
    all_corrupted.extend(corrupted)
    
    # 检查训练标签
    corrupted = check_directory(
        os.path.join(DATA_DIR, "labelsTr"),
        "训练标签"
    )
    all_corrupted.extend(corrupted)
    
    # 检查无标签数据
    corrupted = check_directory(
        os.path.join(DATA_DIR, "imagesUnlabeled"),
        "无标签数据"
    )
    all_corrupted.extend(corrupted)
    
    print(f"\n{'='*60}")
    print(f"检查完成!")
    print(f"{'='*60}")
    print(f"总损坏文件数: {len(all_corrupted)}")
    
    if all_corrupted:
        print("\n建议操作:")
        print("1. 重新下载损坏的文件")
        print("2. 或者从训练中排除这些文件")
        print("\n损坏文件列表:")
        for fpath, _ in all_corrupted:
            print(f"  rm '{fpath}'  # 删除损坏文件")
