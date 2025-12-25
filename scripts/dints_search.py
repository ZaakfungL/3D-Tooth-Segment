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
    DATA_DIR = "/home/ta/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = "./weights"
    ARCH_SAVE_DIR = "./results/dints_arch" 
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(ARCH_SAVE_DIR, exist_ok=True)

    # [Debug 修改] 搜索超参数 - 快速验证模式 (Batch Iteration)
    # Phase 1: Warm-up (仅更新权重)
    WARMUP_STEPS = 1000
    
    # Phase 2: Stabilization (仅更新权重, 进一步稳定)
    ARCH_SEARCH_START_STEPS = 10000
    
    # Phase 3: Joint Optimization (双重更新)
    MAX_ITERATIONS = 20000
    
    EVAL_INTERVAL = 10
    
    BATCH_SIZE = 2
    ROI_SIZE = (96, 96, 96)
    
    # 学习率配置
    LR_WEIGHTS = 0.025     
    LR_ARCH = 3e-4         
    
    # 资源配置
    NUM_WORKERS = 4
    CACHE_RATE = 1       

    # ================= 1. 数据准备 (双层划分) =================
    set_determinism(seed=2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 DiNTS 搜索 | 设备: {device}")
    print(f"👉 策略: Warmup={WARMUP_STEPS} | ArchStart={ARCH_SEARCH_START_STEPS} | Total={MAX_ITERATIONS}")

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

    
    train_loader_w = get_basic_loader(
        data_list=train_files_w, 
        batch_size=BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=True, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        shuffle=False
    )
    
    train_loader_a = get_basic_loader(
        data_list=train_files_a, 
        batch_size=BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=True, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        shuffle=False
    )
    
    val_loader = get_basic_loader(
        data_list=val_files, 
        batch_size=1, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=False, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        shuffle=True
    )

    # ================= 2. 模型与双优化器 =================
    model = DiNTSWrapper(
        in_channels=1, 
        out_channels=2, 
        num_blocks=6,
        num_depths=3,
        channel_mul=1,
        use_downsample=True 
    ).to(device)

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

    # ================= 3. 搜索循环 (Batch Iteration) =================
    best_metric = -1
    best_metric_step = -1
    
    print(f"\n{'='*20} 开始搜索 (Steps: {MAX_ITERATIONS}) {'='*20}")

    global_step = 0
    # 创建无限迭代器辅助函数
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    iter_w = cycle(train_loader_w)
    iter_a = cycle(train_loader_a)

    while global_step < MAX_ITERATIONS:
        global_step += 1
        step_start = time.time()
        model.train()
        
        # 1. 获取数据
        batch_w = next(iter_w)
        batch_a = next(iter_a)

        input_w, label_w = batch_w["image"].to(device), batch_w["label"].to(device)
        input_a, label_a = batch_a["image"].to(device), batch_a["label"].to(device)

        # ------------------------------------------------
        # 阶段 A: 更新架构参数 (Alphas)
        # ------------------------------------------------
        loss_a_val = 0.0
        if global_step > ARCH_SEARCH_START_STEPS:
            optimizer_a.zero_grad()
            output_a = model(input_a)
            loss_a = loss_function(output_a, label_a)
            
            probs_children, _ = model.dints_space.get_prob_a(child=True)
            entropy_loss = model.dints_space.get_topology_entropy(probs_children)
            
            total_loss_a = loss_a + 0.001 * entropy_loss 

            total_loss_a.backward()
            optimizer_a.step()
            loss_a_val = total_loss_a.item()

        # ------------------------------------------------
        # 阶段 B: 更新权重参数 (Weights)
        # ------------------------------------------------
        optimizer_w.zero_grad()
        output_w = model(input_w)
        loss_w = loss_function(output_w, label_w)
        
        loss_w.backward()
        optimizer_w.step()

        step_time = time.time() - step_start
        status_str = "WARMUP" if global_step <= WARMUP_STEPS else \
                     ("STABLE" if global_step <= ARCH_SEARCH_START_STEPS else "SEARCH")
        
        print(f"Step {global_step}/{MAX_ITERATIONS} [{status_str}] | Time: {step_time:.2f}s | "
              f"Loss W: {loss_w.item():.4f} | Loss A: {loss_a_val:.4f}", end="")

        # --- 验证与保存 ---
        if global_step % EVAL_INTERVAL == 0:
            # [显存优化] 验证前先释放训练时的中间变量
            # del input_w, label_w, output_w, loss_w
            # if global_step > ARCH_SEARCH_START_STEPS:
            #     del input_a, label_a
            # torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    # [3090优化] sw_batch_size=1 最小化推理显存峰值
                    val_pred = sliding_window_inference(
                        val_in, ROI_SIZE, sw_batch_size=1, predictor=model,
                        overlap=0.25  # 减少重叠区域，降低计算量
                    )
                    
                    val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_pred)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_pred, y=val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
            
            # 释放验证时的显存
            del val_in, val_lbl, val_pred, val_data
            torch.cuda.empty_cache()
                
            print(f" | Val Dice: {metric:.4f}", end="")
            
            if metric > best_metric:
                best_metric = metric
                best_metric_step = global_step
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
                    
                    print(f" -> 🔥 New Best! Saved arch", end="")
                except Exception as e:
                    print(f" -> [Err] Save Failed: {e}", end="")

        print("") 

    print(f"\n搜索结束。最佳架构 Dice: {best_metric:.4f} (at Step {best_metric_step})")

if __name__ == "__main__":
    try:
        search_baseline()
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()