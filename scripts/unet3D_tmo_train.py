import sys
import os
import glob
import torch
import numpy as np
import itertools 
import time
import warnings 

# 过滤不必要的警告
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

# 导入自定义模块
from src.models.unet3D import UNet3D
from src.dataloaders.basic_loader import get_basic_loader
from src.ssl.utils import update_ema_variables, get_current_consistency_weight, ConsistencyLoss
from src.ssl.tmo import TMOAdamW  # [核心] 导入 TMO 优化器

def train_tmo():
    # ================= 配置区域 =================
    DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = "./weights/ssl_tmo"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 训练超参数
    MAX_EPOCHS = 5
    VAL_INTERVAL = 2
    LR = 1e-4
    ROI_SIZE = (64, 64, 64)
    
    # 比例配置 (Labeled : Unlabeled)
    # TMO 严重依赖有标签梯度的质量，建议 BATCH_SIZE_L 不要太小
    BATCH_SIZE_L = 1
    BATCH_SIZE_U = 1
    
    # Mean Teacher & Consistency 参数
    EMA_DECAY = 0.99       
    CONSISTENCY = 0.1      
    CONSISTENCY_RAMPUP = 20 
    
    # 资源配置
    NUM_WORKERS = 0
    CACHE_RATE = 0.0

    # ================= 1. 数据准备 =================
    set_determinism(seed=2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 TMO (Trusted Momentum) 训练 | 设备: {device}")

    # A. 有标签数据
    labeled_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labeled_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    labeled_dicts = [{"image": i, "label": l} for i, l in zip(labeled_images, labeled_labels)]
    
    # 划分 Train/Val
    train_labeled_files, val_files = partition_dataset(
        data=labeled_dicts, ratios=[0.8, 0.2], shuffle=True, seed=2025
    )

    # B. 无标签数据
    unlabeled_dir = os.path.join(DATA_DIR, "imagesUnlabeled")
    if os.path.exists(unlabeled_dir):
        unlabeled_images = sorted(glob.glob(os.path.join(unlabeled_dir, "*.nii.gz")))
        unlabeled_dicts = [{"image": i, "label": i} for i in unlabeled_images]
        print(f"  - 有标签数据: {len(train_labeled_files)}")
        print(f"  - 无标签数据: {len(unlabeled_dicts)}")
    else:
        print("❌ 警告: 未找到 imagesUnlabeled，TMO 将回退到伪SSL模式！")
        unlabeled_dicts = train_labeled_files 

    # C. 创建加载器
    loader_labeled = get_basic_loader(
        data_list=train_labeled_files,
        batch_size=BATCH_SIZE_L, 
        roi_size=ROI_SIZE, 
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        limit=1
    )
    
    loader_unlabeled = get_basic_loader(
        data_list=unlabeled_dicts,
        batch_size=BATCH_SIZE_U, 
        roi_size=ROI_SIZE, 
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        limit=1
    )
    
    loader_val = get_basic_loader(
        data_list=val_files,
        batch_size=1, 
        roi_size=ROI_SIZE, 
        is_train=False, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        limit=1
    )

    # ================= 2. 模型与优化器 =================
    def create_model():
        return UNet3D(in_channels=1, out_channels=2).to(device)

    model = create_model()      # Student
    ema_model = create_model()  # Teacher (EMA)

    # 初始化 Teacher
    for param in ema_model.parameters():
        param.detach_()
    ema_model.load_state_dict(model.state_dict())

    # 损失函数
    loss_supervised = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_consistency = ConsistencyLoss()
    
    # [核心] 使用 TMO 优化器
    optimizer = TMOAdamW(model.parameters(), lr=LR)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 3. 训练循环 =================
    best_metric = -1
    
    print(f"\n{'='*20} Start TMO Training {'='*20}")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        ema_model.train()
        
        loss_sup_sum = 0
        loss_cons_sum = 0
        step = 0
        
        consistency_weight = get_current_consistency_weight(epoch, MAX_EPOCHS, CONSISTENCY, CONSISTENCY_RAMPUP)
        
        # 循环迭代
        train_iterator = zip(loader_unlabeled, itertools.cycle(loader_labeled))
        
        for batch_u, batch_l in train_iterator:
            step += 1
            
            # 数据准备
            img_l, lbl_l = batch_l["image"].to(device), batch_l["label"].to(device)
            img_u = batch_u["image"].to(device)
            
            # -------------------------------------------------------
            # [步骤 1] Labeled Step: 建立可信方向 (Trusted Direction)
            # -------------------------------------------------------
            optimizer.zero_grad() # 清空梯度
            
            pred_l = model(img_l)
            loss_sup = loss_supervised(pred_l, lbl_l)
            
            # 反向传播产生 g_L
            loss_sup.backward()
            optimizer.step_labeled()
            
            # -------------------------------------------------------
            # [步骤 2] Unlabeled Step: 谨慎更新 (Cautious Update)
            # -------------------------------------------------------
            optimizer.zero_grad() # 清空 g_L，准备计算 g_U
            
            # Student Forward
            pred_u_student = model(img_u)
            # Teacher Forward (No Grad)
            with torch.no_grad():
                pred_u_teacher = ema_model(img_u)
            
            # 计算一致性损失
            # 对 Teacher 做 Sharpening (可选，但推荐)
            teacher_prob = torch.softmax(pred_u_teacher, dim=1)
            student_prob = torch.softmax(pred_u_student, dim=1)
            
            loss_cons = loss_consistency(student_prob, teacher_prob)
            total_loss_cons = consistency_weight * loss_cons

            # 反向传播产生 g_U
            total_loss_cons.backward()
            optimizer.step_unlabeled()

            # -------------------------------------------------------
            # [步骤 3] Teacher EMA 更新
            # -------------------------------------------------------
            update_ema_variables(model, ema_model, EMA_DECAY, epoch * 100 + step)
            
            loss_sup_sum += loss_sup.item()
            loss_cons_sum += loss_cons.item()

        # 日志记录
        epoch_time = time.time() - epoch_start
        print(f"Ep {epoch+1}/{MAX_EPOCHS} | Time: {epoch_time:.0f}s | "
              f"L_Sup: {loss_sup_sum/max(step,1):.4f} | "
              f"L_Cons (w={consistency_weight:.3f}): {loss_cons_sum/max(step,1):.4f}", end="")

        # --- Validation (使用 Teacher 评估) ---
        if (epoch + 1) % VAL_INTERVAL == 0:
            ema_model.eval()
            with torch.no_grad():
                for val_data in loader_val:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    # 滑动窗口推理
                    val_out = sliding_window_inference(val_in, ROI_SIZE, 4, ema_model)
                    
                    val_out = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_out)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    
                    dice_metric(y_pred=val_out, y=val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f" | Val Dice: {metric:.4f}", end="")
                
                if metric > best_metric:
                    best_metric = metric
                    # torch.save(ema_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_teacher_tmo.pth"))
                    # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_student_tmo.pth"))
                    print(f" -> 🔥 Saved!", end="")

        print("")

    print(f"训练结束。Best Dice: {best_metric:.4f}")

if __name__ == "__main__":
    try:
        train_tmo()
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()