import sys
import os
import glob
import torch
import json
import warnings
import numpy as np
import time
import warnings

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
    GPU_ID = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    print(f"使用GPU: {GPU_ID}")

    DATA_DIR = "/home/ta/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = "./weights"
    ARCH_SAVE_DIR = "./results/dints_arch" 
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(ARCH_SAVE_DIR, exist_ok=True)

    # ============ 论文 Search 阶段参数 (严格对应) ============
    # 论文: "We train w for the first 1k warm-up and following 10k iterations 
    #        without updating architecture. In the following 10k iterations,
    #        we jointly optimize w with SGD and α,pe with Adam"
    
    # Phase 1: Warm-up (仅更新权重, LR从0.025线性增到0.2)
    WARMUP_STEPS = 1000
    
    # Phase 2: Stabilization (仅更新权重, 共10k步)
    # 架构搜索开始于 1k + 10k = 11k
    ARCH_SEARCH_START_STEPS = 11000
    
    # Phase 3: Joint Optimization (双重更新, 共10k步)
    # 总迭代: 1k + 10k + 10k = 21k (论文原值，对应batch=8)
    MAX_ITERATIONS = 21000
    EVAL_INTERVAL = 100
    
    # LR Decay 节点 (论文: "decays with factor 0.5 at [8k, 16k] iterations")
    LR_DECAY_STEPS = [8000, 16000]
    LR_DECAY_FACTOR = 0.5
    
    BATCH_SIZE = 1
    NUM_SAMPLES = 1
    ROI_SIZE = (96, 96, 96)
    
    # 学习率配置 (论文原值)
    LR_WEIGHTS_INIT = 0.025
    LR_WEIGHTS_MAX = 0.2
    LR_ARCH = 0.008         
    
    # 资源配置
    NUM_WORKERS = 3
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
        num_samples=NUM_SAMPLES,
        is_train=True, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        shuffle=True
    )
    
    train_loader_a = get_basic_loader(
        data_list=train_files_a, 
        batch_size=BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        num_samples=NUM_SAMPLES,
        is_train=True, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        shuffle=True
    )
    
    val_loader = get_basic_loader(
        data_list=val_files, 
        batch_size=1, 
        roi_size=ROI_SIZE, 
        num_samples=1,
        is_train=False, 
        num_workers=NUM_WORKERS, 
        cache_rate=CACHE_RATE,
        shuffle=False
    )

    # ================= 2. 模型与双优化器 =================
    model = DiNTSWrapper(
        in_channels=1, 
        out_channels=2, 
        num_blocks=12,
        num_depths=4,
        channel_mul=1,
        use_downsample=True 
    ).to(device)

    if hasattr(model, "dints_space"):
        model.dints_space.device = device

    optimizer_w = torch.optim.SGD(
        model.weight_parameters(), 
        lr=LR_WEIGHTS_INIT,
        momentum=0.9, 
        weight_decay=4e-5
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
    
    loop_start_time = time.time()

    while global_step < MAX_ITERATIONS:
        global_step += 1
        model.train()
        
        # ------------------------------------------------
        # 学习率调度
        # ------------------------------------------------
        if global_step <= WARMUP_STEPS:
            # Phase 1: Linear Warmup (0.025 -> 0.2)
            lr = LR_WEIGHTS_INIT + (LR_WEIGHTS_MAX - LR_WEIGHTS_INIT) * (global_step / WARMUP_STEPS)
        else:
            # Phase 2 & 3: Step Decay
            lr = LR_WEIGHTS_MAX
            for decay_step in LR_DECAY_STEPS:
                if global_step >= decay_step:
                    lr *= LR_DECAY_FACTOR
        
        # 更新优化器学习率
        for param_group in optimizer_w.param_groups:
            param_group['lr'] = lr
        
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
        current_time = time.time()
        step_time = current_time - loop_start_time
        loop_start_time = current_time
        
        status_str = "WARMUP" if global_step <= WARMUP_STEPS else \
                     ("STABLE" if global_step <= ARCH_SEARCH_START_STEPS else "SEARCH")
        
        if global_step % 10 == 0:
            print(f"Step {global_step}/{MAX_ITERATIONS} [{status_str}] | Time: {step_time:.2f}s | "
                  f"Loss W: {loss_w.item():.4f} | Loss A: {loss_a_val:.4f}")

        # --- 验证与保存 ---
        if global_step % EVAL_INTERVAL == 0:
            torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    val_pred = sliding_window_inference(
                        val_in, ROI_SIZE, 
                        sw_batch_size=8,  # 推理显存优化
                        predictor=model,
                        overlap=0.5
                    )
                    
                    val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_pred)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_pred, y=val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
            
            print(f"Validation at Step {global_step} | Val Dice: {metric:.4f}", end="")
            
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
            
            # [显存优化] 验证后释放显存
            torch.cuda.empty_cache()
            model.train() 

    print(f"\n搜索结束。最佳架构 Dice: {best_metric:.4f} (at Step {best_metric_step})")

if __name__ == "__main__":
    try:
        search_baseline()
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()