import sys
import os
import glob
import torch
import numpy as np
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
    # GPU配置 - 指定使用哪张显卡
    GPU_ID = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    print(f"使用GPU: {GPU_ID}")

    DATA_DIR = "/home/ta/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = "./weights/ssl_tmo"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Batch 配置
    LOAD_BATCH_SIZE_L = 1
    NUM_SAMPLES_L = 16
    
    LOAD_BATCH_SIZE_U = 1
    NUM_SAMPLES_U = 16
    
    # 用于控制RAM读入数据量
    NUM_LABELED_USE = 18        # Labeled 数据量
    NUM_UNLABELED_USE = 18      # Unlabeled 数据量
    
    # 训练超参数（基于iteration）
    MAX_ITERATIONS = 5400  # 最大迭代次数
    VAL_INTERVAL = 90      # 验证间隔
    
    LR = 1e-4
    ROI_SIZE = (96, 96, 96)
    
    # Mean Teacher & Consistency 参数
    EMA_DECAY = 0.99       
    CONSISTENCY = 0.1      
    CONSISTENCY_RAMPUP = MAX_ITERATIONS // 5  # 前 20% 的时间用于 Rampup
    
    # 资源配置
    NUM_WORKERS = 3
    CACHE_RATE = 1.0

    # ================= 1. 数据准备 =================
    set_determinism(seed=2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 TMO (Trusted Momentum) 训练 | 设备: {device}")
    print(f"📌 总计 {MAX_ITERATIONS} Iterations")

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
        
        import random
        random.shuffle(unlabeled_dicts)
        unlabeled_dicts = unlabeled_dicts[:NUM_UNLABELED_USE]
        print(f"⚠️ 已限制无标签数据量: {len(unlabeled_images)} -> {len(unlabeled_dicts)}")
            
        print(f"  - 有标签数据 (Train): {len(train_labeled_files)} 例")
        print(f"  - 无标签数据 (Train): {len(unlabeled_dicts)} 例")
    else:
        print("❌ 错误: 未找到 imagesUnlabeled 文件夹，无法进行半监督训练！")
        print(f"   请确保目录存在: {unlabeled_dir}")
        sys.exit(1) 

    # C. 创建加载器
    # 1. 有标签加载器
    loader_labeled = get_basic_loader(
        data_list=train_labeled_files,
        batch_size=LOAD_BATCH_SIZE_L, 
        roi_size=ROI_SIZE, 
        num_samples=NUM_SAMPLES_L,
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
    )
    
    # 2. 无标签加载器
    loader_unlabeled = get_basic_loader(
        data_list=unlabeled_dicts,
        batch_size=LOAD_BATCH_SIZE_U, 
        roi_size=ROI_SIZE, 
        num_samples=NUM_SAMPLES_U,
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
    )
    
    # 3. 验证加载器
    loader_val = get_basic_loader(
        data_list=val_files,
        batch_size=1, 
        roi_size=ROI_SIZE, 
        is_train=False, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
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

    # 损失函数
    loss_supervised = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_consistency = ConsistencyLoss()
    
    # [核心差异] 使用 TMO 优化器
    optimizer = TMOAdamW(model.parameters(), lr=LR)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 3. 训练循环 (Iteration Based) =================
    best_metric = -1
    best_metric_iter = -1
    iteration = 0
    
    print(f"\n{'='*20} Start TMO Training (Iteration Based) {'='*20}")
    
    model.train()
    ema_model.train()
    
    iter_labeled = iter(loader_labeled)
    iter_unlabeled = iter(loader_unlabeled)
    
    start_time = time.time()
    loop_start_time = time.time()

    while iteration < MAX_ITERATIONS:
        # 数据获取 (无限循环)
        try:
            batch_l = next(iter_labeled)
        except StopIteration:
            iter_labeled = iter(loader_labeled)
            batch_l = next(iter_labeled)
            
        try:
            batch_u = next(iter_unlabeled)
        except StopIteration:
            iter_unlabeled = iter(loader_unlabeled)
            batch_u = next(iter_unlabeled)
            
        iteration += 1
        
        img_l, lbl_l = batch_l["image"].to(device), batch_l["label"].to(device)
        img_u = batch_u["image"].to(device)
        
        consistency_weight = get_current_consistency_weight(iteration, MAX_ITERATIONS, CONSISTENCY, CONSISTENCY_RAMPUP)
        
        # -------------------------------------------------------
        # [TMO 步骤 1] Labeled Step: 建立可信方向 (Trusted Direction)
        # -------------------------------------------------------
        optimizer.zero_grad()
        
        pred_l = model(img_l)
        l_sup = loss_supervised(pred_l, lbl_l)
        
        # 反向传播产生 g_L
        l_sup.backward()
        optimizer.step_labeled()  # [TMO 特有] 保存可信梯度方向
        
        # -------------------------------------------------------
        # [TMO 步骤 2] Unlabeled Step: 谨慎更新 (Cautious Update)
        # -------------------------------------------------------
        optimizer.zero_grad()
        
        # Student Forward
        pred_u_student = model(img_u)
        # Teacher Forward (No Grad)
        with torch.no_grad():
            pred_u_teacher = ema_model(img_u)
        
        # 计算一致性损失
        student_prob = torch.softmax(pred_u_student, dim=1)
        teacher_prob = torch.softmax(pred_u_teacher, dim=1)
        l_cons = loss_consistency(student_prob, teacher_prob)
        total_loss_cons = consistency_weight * l_cons

        # 反向传播产生 g_U
        total_loss_cons.backward()
        optimizer.step_unlabeled()  # [TMO 特有] 根据 g_L 过滤 g_U
        
        # -------------------------------------------------------
        # [步骤 3] Teacher EMA 更新
        # -------------------------------------------------------
        update_ema_variables(model, ema_model, EMA_DECAY, iteration)
        
        # --- Logging ---
        current_time = time.time()
        iter_time = current_time - loop_start_time
        loop_start_time = current_time
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}/{MAX_ITERATIONS} | Time: {iter_time:.4f}s | "
                  f"L_Sup: {l_sup.item():.4f} | L_Cons: {l_cons.item():.4f} (w={consistency_weight:.3f})")

        # --- Validation (使用 Teacher 评估) ---
        if iteration % VAL_INTERVAL == 0:
            torch.cuda.empty_cache()
            
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
                
                print(f"Validation at Iter {iteration} | Val Dice: {metric:.4f}", end="")
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_iter = iteration
                    # torch.save(ema_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_teacher_tmo.pth"))
                    # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_student_tmo.pth"))
                    print(f" -> 🔥 New Best! ({best_metric:.4f})")
                else:
                    print("")
            
            torch.cuda.empty_cache()
            
            model.train()
            ema_model.train()

    total_time = time.time() - start_time
    print(f"\n训练结束。总用时: {total_time:.1f}s")
    print(f"最佳模型 Dice: {best_metric:.4f} 于 Iteration {best_metric_iter}")

if __name__ == "__main__":
    try:
        train_tmo()
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()