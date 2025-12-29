import sys
import os
import glob
import torch
import time
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module="monai.inferers.utils")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch, partition_dataset
from monai.transforms import AsDiscrete

from src.models.unet3D import UNet3D
from src.dataloaders.basic_loader import get_basic_loader
from src.ssl.utils import update_ema_variables, get_current_consistency_weight, ConsistencyLoss

def train_ssl(seed=2025):
    # ================= 配置区域 =================
    # GPU配置 - 指定使用哪张显卡
    GPU_ID = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    print(f"使用GPU: {GPU_ID}")

    DATA_DIR = "/home/ta/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    MODEL_SAVE_DIR = f"./weights/ssl_meanteacher_seed{seed}"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    LOAD_BATCH_SIZE_L = 1
    NUM_SAMPLES_L = 16
    
    LOAD_BATCH_SIZE_U = 1
    NUM_SAMPLES_U = 16
    
    # 用于控制RAM读入数据量
    NUM_LABELED_USE = 18        # Labeled 数据量
    NUM_UNLABELED_USE = 18      # Unlabeled 数据量
    
    # 训练超参数（基于iteration）
    MAX_ITERATIONS = 7200  # 最大迭代次数
    VAL_INTERVAL = 90      # 验证间隔
    
    LR = 1e-4
    ROI_SIZE = (96, 96, 96)
    
    # Mean Teacher 参数
    EMA_DECAY = 0.99       
    CONSISTENCY = 0.1      
    CONSISTENCY_RAMPUP = MAX_ITERATIONS // 5 # 前 20% 的时间用于 Rampup
    
    # 资源配置
    NUM_WORKERS = 3
    CACHE_RATE = 1.0

    # ================= 1. 数据准备 =================
    set_determinism(seed=seed)  # [修改] 使用传入的seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 Mean Teacher 训练 | 设备: {device}")
    print(f"📌 总计 {MAX_ITERATIONS} Iterations")
    print(f"当前随机种子: {seed}")

    # A. 准备有标签数据 (Labeled)
    labeled_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labeled_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    labeled_dicts = [{"image": i, "label": l} for i, l in zip(labeled_images, labeled_labels)]
    
    # 划分 Train/Val
    train_labeled_files, val_files = partition_dataset(
        data=labeled_dicts, ratios=[0.8, 0.2], shuffle=True, seed=seed  # [修改] 使用传入的seed
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

    loss_supervised = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_consistency = ConsistencyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 3. 训练循环 =================
    best_metric = -1
    best_metric_iter = -1
    iteration = 0
    
    print(f"\n{'='*20} Start Training (Iteration Based) {'='*20}")
    
    model.train()
    ema_model.train()
    
    iter_labeled = iter(loader_labeled)
    iter_unlabeled = iter(loader_unlabeled)
    
    start_time = time.time()
    loop_start_time = time.time()

    while iteration < MAX_ITERATIONS:
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
        
        optimizer.zero_grad()
        
        # --- Forward ---
        pred_l_student = model(img_l)
        pred_u_student = model(img_u)

        with torch.no_grad():
            pred_u_teacher = ema_model(img_u)
        
        # --- Loss ---
        # Labeled Loss
        l_sup = loss_supervised(pred_l_student, lbl_l)
        
        # Unlabeled Loss (Consistency)
        student_prob = torch.softmax(pred_u_student, dim=1)
        teacher_prob = torch.softmax(pred_u_teacher, dim=1)
        l_cons = loss_consistency(student_prob, teacher_prob)
        
        total_loss = l_sup + consistency_weight * l_cons

        # --- Backward ---
        total_loss.backward()
        optimizer.step()
        
        # --- EMA Update ---
        update_ema_variables(model, ema_model, EMA_DECAY, iteration)
        
        # --- Logging ---
        current_time = time.time()
        iter_time = current_time - loop_start_time
        loop_start_time = current_time
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}/{MAX_ITERATIONS} | Time: {iter_time:.4f}s | "
                  f"L_Sup: {l_sup.item():.4f} | L_Cons: {l_cons.item():.4f} (w={consistency_weight:.3f})")

        # --- Validation ---
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
                    save_path = os.path.join(MODEL_SAVE_DIR, "best_unet3D_ssl_model.pth")
                    # torch.save(ema_model.state_dict(), save_path)
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
    parser = argparse.ArgumentParser(description="UNet3D Mean Teacher 半监督训练脚本")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子 (默认: 2025)")
    args = parser.parse_args()
    
    try:
        train_ssl(seed=args.seed)
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()