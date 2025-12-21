import sys
import os
import glob
import torch
import json
import warnings
import numpy as np
import time
import warnings

# [新增] 忽略来自 MONAI/PyTorch 的特定未来警告，保持日志干净
warnings.filterwarnings("ignore", category=UserWarning, module="monai.inferers.utils")


# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch, partition_dataset
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

# 导入你的模块
from src.models.dints import DiNTSWrapper
from src.dataloaders.basic_loader import get_basic_loader

def search_baseline():
    # ================= 配置区域 =================
    DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = "./weights"
    ARCH_SAVE_DIR = "./results/dints_arch" 
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(ARCH_SAVE_DIR, exist_ok=True)

    # [Debug 修改] 搜索超参数 - 快速验证模式
    MAX_EPOCHS = 5         
    VAL_INTERVAL = 1       
    
    # ⚠️ 显存优化关键点：
    # 虽然这里设为 1，但由于 basic_loader 里 num_samples=2，实际 Batch 是 2
    BATCH_SIZE = 1         
    ROI_SIZE = (64, 64, 64)
    
    # 学习率配置
    LR_WEIGHTS = 0.025     
    LR_ARCH = 3e-4         
    
    # 资源配置
    NUM_WORKERS = 2
    CACHE_RATE = 0.0       

    # ================= 1. 数据准备 (双层划分) =================
    set_determinism(seed=2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 DiNTS 搜索 (极简模式) | 设备: {device}")

    images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]

    # NAS 数据划分
    train_files_all, val_files = partition_dataset(
        data=data_dicts, ratios=[0.8, 0.2], shuffle=True, seed=2025
    )
    
    train_files_w, train_files_a = partition_dataset(
        data=train_files_all, ratios=[0.5, 0.5], shuffle=True, seed=2025
    )

    print(f"  - 总数据: {len(data_dicts)}")
    print(f"  - 权重更新集 (Train_W): {len(train_files_w)}")
    print(f"  - 架构更新集 (Train_A): {len(train_files_a)}")
    print(f"  - 验证集 (Val): {len(val_files)}")

    # [Debug] 限制数据量
    print("⚡ 正在创建加载器 (限制数据量为 2)...")
    
    train_loader_w = get_basic_loader(
        data_list=train_files_w, 
        batch_size=BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=True, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        limit=1
    )
    
    train_loader_a = get_basic_loader(
        data_list=train_files_a, 
        batch_size=BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=True, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        limit=1
    )
    
    val_loader = get_basic_loader(
        data_list=val_files, 
        batch_size=1, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=False, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        limit=1
    )

    # ================= 2. 模型与双优化器 =================
    # [显存优化] 大幅削减通道数和层数，确保 8G 显存能跑通
    print("🔧 初始化 DiNTS 模型 (channel_mul=0.25, num_blocks=4)...")
    model = DiNTSWrapper(
        in_channels=1, 
        out_channels=2, 
        num_blocks=4,      # [修改] 从 6 降到 4
        num_depths=3,
        channel_mul=0.25,  # [修改] 从 0.5 降到 0.25 (通道数减半)
        use_downsample=True 
    ).to(device)

    # [关键修复] 显式同步 TopologySearch 内部的 device 属性
    if hasattr(model, "dints_space"):
        model.dints_space.device = device

    optimizer_w = torch.optim.SGD(
        model.weight_parameters(), 
        lr=LR_WEIGHTS, 
        momentum=0.9, 
        weight_decay=3e-4
    )
    
    optimizer_a = torch.optim.Adam(
        model.arch_parameters(), 
        lr=LR_ARCH, 
        betas=(0.5, 0.999), 
        weight_decay=0
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 3. 搜索循环 =================
    best_metric = -1
    best_metric_epoch = -1
    
    print(f"\n{'='*20} 开始搜索 {'='*20}")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        loss_w_sum = 0
        loss_a_sum = 0
        step = 0
        
        # 使用 zip 同时遍历两个加载器
        for batch_w, batch_a in zip(train_loader_w, train_loader_a):
            step += 1
            
            input_w, label_w = batch_w["image"].to(device), batch_w["label"].to(device)
            input_a, label_a = batch_a["image"].to(device), batch_a["label"].to(device)

            # ------------------------------------------------
            # 阶段 A: 更新架构参数 (Alphas)
            # ------------------------------------------------
            optimizer_a.zero_grad()
            output_a = model(input_a)
            loss_a = loss_function(output_a, label_a)
            
            # [Fix 核心修复: IndexKernel Error] 
            probs_children, _ = model.dints_space.get_prob_a(child=True)
            entropy_loss = model.dints_space.get_topology_entropy(probs_children)
            
            total_loss_a = loss_a + 0.001 * entropy_loss 

            total_loss_a.backward()
            optimizer_a.step()
            loss_a_sum += total_loss_a.item()

            # ------------------------------------------------
            # 阶段 B: 更新权重参数 (Weights)
            # ------------------------------------------------
            optimizer_w.zero_grad()
            output_w = model(input_w)
            loss_w = loss_function(output_w, label_w)
            
            loss_w.backward()
            optimizer_w.step()
            loss_w_sum += loss_w.item()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Time: {epoch_time:.1f}s | "
              f"Loss W: {loss_w_sum/max(step,1):.4f} | Loss A: {loss_a_sum/max(step,1):.4f}", end="")

        # --- 验证与保存 ---
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    val_pred = sliding_window_inference(val_in, ROI_SIZE, 4, model)
                    
                    val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_pred)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_pred, y=val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f" | Val Dice: {metric:.4f}", end="")
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    try:
                        # 获取最佳架构
                        topology = model.get_topology()
                        
                        arch_json = {
                            "arch_code_a": topology[1].tolist(),
                            "arch_code_c": topology[2].tolist()
                        }
                        save_path = os.path.join(ARCH_SAVE_DIR, "best_arch_roi.json")
                        with open(save_path, "w") as f:
                            json.dump(arch_json, f, indent=4)
                        
                        # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "dints_search_best.pth"))
                        print(f" -> 🔥 New Best! Saved arch", end="")
                    except Exception as e:
                        print(f" -> [Err] Save Failed: {e}", end="")

        print("") 

    print(f"\n搜索结束。最佳架构 Dice: {best_metric:.4f}")

if __name__ == "__main__":
    try:
        search_baseline()
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()