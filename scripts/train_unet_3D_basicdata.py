import sys
import os
import glob
import torch
import time
import warnings # [新增]


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

def train_baseline():
    # ================= 配置区域 =================
    # 路径配置
    DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"  # 你的 ROI 数据路径
    MODEL_SAVE_DIR = "./models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 训练超参数
    MAX_EPOCHS = 100
    VAL_INTERVAL = 2        # 每多少个 epoch 验证一次
    BATCH_SIZE = 2
    LR = 1e-4
    ROI_SIZE = (96, 96, 96) # Patch 大小
    
    # 显存/内存优化配置
    AMP = True              # 开启混合精度
    NUM_WORKERS = 2         # WSL建议设为2或0
    CACHE_RATE = 0.0        # 设为0.0防止内存溢出
    
    # ================= 1. 数据准备 =================
    set_determinism(seed=2025) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("正在扫描并划分数据集...")
    images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    
    if not images:
        raise ValueError(f"错误：在 {DATA_DIR} 中未找到数据！")
        
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    train_files, val_files = partition_dataset(
        data=data_dicts, 
        ratios=[0.8, 0.2], 
        shuffle=True, 
        seed=2025
    )
    
    print(f"  - 总数据量: {len(data_dicts)}")
    print(f"  - 训练集 (80%): {len(train_files)} 例")
    print(f"  - 验证集 (20%): {len(val_files)} 例")

    # ================= 2. 创建加载器 =================
    train_loader = get_basic_loader(
        data_list=train_files,
        batch_size=BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE
    )
    
    val_loader = get_basic_loader(
        data_list=val_files,
        batch_size=1,
        roi_size=ROI_SIZE, 
        is_train=False, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE
    )

    # ================= 3. 模型与优化器 =================
    model = UNet3D(in_channels=1, out_channels=2).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if AMP else None

    # ================= 4. 训练循环 =================
    best_metric = -1
    best_metric_epoch = -1
    
    print(f"\n{'='*20} 开始训练 {'='*20}")
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        step = 0
        
        # --- Training (无 tqdm) ---
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            
            if AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss /= step
        epoch_time = time.time() - epoch_start
        
        # 打印训练日志
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS} | Time: {epoch_time:.1f}s | Train Loss: {epoch_loss:.4f}", end="")

        # --- Validation (无 tqdm) ---
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, 
                        roi_size=ROI_SIZE, 
                        sw_batch_size=4, 
                        predictor=model
                    )
                    
                    val_outputs = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_outputs)]
                    val_labels = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                print(f" | Val Dice: {metric:.4f}", end="")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join(MODEL_SAVE_DIR, "best_metric_model.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f" -> 🔥 New Best! ({best_metric:.4f})", end="")
        
        # 换行，为下一个 Epoch 做准备
        print("") 

    print(f"\n训练结束。最佳模型 Dice: {best_metric:.4f} 于 Epoch {best_metric_epoch}")

if __name__ == "__main__":
    try:
        train_baseline()
    except Exception as e:
        print(f"❌ 训练发生错误: {e}")
        import traceback
        traceback.print_exc()