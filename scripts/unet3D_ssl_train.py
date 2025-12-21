import sys
import os
import glob
import torch
import numpy as np
import itertools # [新增] 用于循环数据集
from tqdm import tqdm
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
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch, partition_dataset
from monai.transforms import AsDiscrete

# 导入你的模块
from src.models.unet3D import UNet3D
from src.dataloaders.basic_loader import get_basic_loader
from src.ssl.utils import update_ema_variables, get_current_consistency_weight, ConsistencyLoss

def train_ssl():
    # ================= 配置区域 =================
    DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = "./weights/ssl_meanteacher"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # SSL 超参数
    MAX_EPOCHS = 5
    VAL_INTERVAL = 2
    LR = 1e-4
    ROI_SIZE = (64, 64, 64)
    
    # [核心修改] 比例控制区域
    # 这里控制一个 Batch 内 "有标签:无标签" 的数量比例
    # 显存占用 ≈ (BATCH_SIZE_L + BATCH_SIZE_U) * 显存消耗
    # 建议: 保持 1:1 (2 vs 2) 或 1:2 (1 vs 2) 防止显存爆炸
    BATCH_SIZE_L = 1  # 有标签 Batch Size
    BATCH_SIZE_U = 1  # 无标签 Batch Size (增大此值可实现 1:N)
    
    # Mean Teacher 参数
    EMA_DECAY = 0.99       
    CONSISTENCY = 0.1      
    CONSISTENCY_RAMPUP = 20 
    
    # 资源配置
    AMP = True
    NUM_WORKERS = 2
    CACHE_RATE = 0.0

    # ================= 1. 数据准备 =================
    set_determinism(seed=2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 Mean Teacher 训练 | 设备: {device}")
    print(f"📌 Batch 比例配置: Labeled={BATCH_SIZE_L} : Unlabeled={BATCH_SIZE_U}")

    # A. 准备有标签数据 (Labeled)
    labeled_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labeled_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    labeled_dicts = [{"image": i, "label": l} for i, l in zip(labeled_images, labeled_labels)]
    
    # 划分 Train/Val
    train_labeled_files, val_files = partition_dataset(
        data=labeled_dicts, ratios=[0.8, 0.2], shuffle=True, seed=2025
    )

    # B. 准备无标签数据 (Unlabeled)
    unlabeled_dir = os.path.join(DATA_DIR, "imagesUnlabeled")
    if os.path.exists(unlabeled_dir):
        unlabeled_images = sorted(glob.glob(os.path.join(unlabeled_dir, "*.nii.gz")))
        unlabeled_dicts = [{"image": i, "label": i} for i in unlabeled_images]
        print(f"  - 有标签数据 (Train): {len(train_labeled_files)} 例")
        print(f"  - 无标签数据 (Train): {len(unlabeled_dicts)} 例")
    else:
        print("❌ 警告: 未找到 imagesUnlabeled 文件夹，回退到纯监督模式！")
        unlabeled_dicts = train_labeled_files 

    # C. 创建加载器
    # 1. 有标签加载器 (使用 BATCH_SIZE_L)
    loader_labeled = get_basic_loader(
        data_list=train_labeled_files,
        batch_size=BATCH_SIZE_L, 
        roi_size=ROI_SIZE, 
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        limit=1
    )
    
    # 2. 无标签加载器 (使用 BATCH_SIZE_U)
    loader_unlabeled = get_basic_loader(
        data_list=unlabeled_dicts,
        batch_size=BATCH_SIZE_U, 
        roi_size=ROI_SIZE, 
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        limit=1
    )
    
    # 3. 验证加载器
    loader_val = get_basic_loader(
        data_list=val_files,
        batch_size=1, 
        roi_size=ROI_SIZE, 
        is_train=False, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        limit=1
    )

    # ================= 2. 模型初始化 =================
    def create_model():
        model = UNet3D(in_channels=1, out_channels=2).to(device)
        return model

    model = create_model()          # Student
    ema_model = create_model()      # Teacher

    for param in ema_model.parameters():
        param.detach_() 
    ema_model.load_state_dict(model.state_dict())

    loss_supervised = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_consistency = ConsistencyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if AMP else None
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 3. 训练循环 =================
    best_metric = -1
    
    print(f"\n{'='*20} Start Training {'='*20}")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        ema_model.train()
        
        loss_sup_sum = 0
        loss_cons_sum = 0
        step = 0
        
        consistency_weight = get_current_consistency_weight(epoch, MAX_EPOCHS, CONSISTENCY, CONSISTENCY_RAMPUP)
        
        # [核心修改] 循环策略
        # 使用 zip(loader_unlabeled, itertools.cycle(loader_labeled))
        # 1. 以 loader_unlabeled (大数据集) 的长度为准，保证每个 Epoch 遍历完所有无标签数据
        # 2. loader_labeled (小数据集) 会无限循环，直到无标签数据跑完
        # 3. 这样实现了 "1个Epoch内，所有无标签数据被训练1次，有标签数据被重复训练多次"
        
        train_iterator = zip(loader_unlabeled, itertools.cycle(loader_labeled))
        
        for batch_u, batch_l in train_iterator:
            step += 1
            
            img_l, lbl_l = batch_l["image"].to(device), batch_l["label"].to(device)
            img_u = batch_u["image"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=AMP):
                # 1. Forward
                pred_l_student = model(img_l)
                pred_u_student = model(img_u)

                with torch.no_grad():
                    pred_u_teacher = ema_model(img_u)
                
                # 2. Loss
                # Labeled Loss
                l_sup = loss_supervised(pred_l_student, lbl_l)
                
                # Unlabeled Loss (Consistency)
                student_prob = torch.softmax(pred_u_student, dim=1)
                teacher_prob = torch.softmax(pred_u_teacher, dim=1)
                l_cons = loss_consistency(student_prob, teacher_prob)
                
                total_loss = l_sup + consistency_weight * l_cons

            # 3. Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 4. EMA Update
            # 使用全局步数 (epoch * steps_per_epoch + step) 可能会因为 steps 变化而不准
            # 这里简单累加即可
            update_ema_variables(model, ema_model, EMA_DECAY, epoch * 100 + step)
            
            loss_sup_sum += l_sup.item()
            loss_cons_sum += l_cons.item()

        epoch_time = time.time() - epoch_start
        print(f"Ep {epoch+1}/{MAX_EPOCHS} | Time: {epoch_time:.0f}s | Steps: {step} | "
              f"L_Sup: {loss_sup_sum/max(step,1):.4f} | "
              f"L_Cons (w={consistency_weight:.3f}): {loss_cons_sum/max(step,1):.4f}", end="")

        # --- Validation ---
        if (epoch + 1) % VAL_INTERVAL == 0:
            ema_model.eval()
            with torch.no_grad():
                for val_data in loader_val:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    val_out = sliding_window_inference(val_in, ROI_SIZE, 4, ema_model)
                    val_out = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_out)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_out, y=val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f" | Val Dice: {metric:.4f}", end="")
                
                if metric > best_metric:
                    best_metric = metric
                    # torch.save(ema_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_teacher.pth"))
                    print(f" -> 🔥 Saved!", end="")

        print("")

    print(f"训练结束。Best Dice: {best_metric:.4f}")

if __name__ == "__main__":
    train_ssl()