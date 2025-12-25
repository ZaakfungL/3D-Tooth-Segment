import sys
import os
import glob
import torch
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

def train_baseline():
    # ================= 配置区域 =================
    # GPU配置 - 指定使用哪张显卡
    GPU_ID = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    print(f"使用GPU: {GPU_ID}")
    
    # 路径配置
    DATA_DIR = "/home/ta/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"  # 你的 ROI 数据路径
    MODEL_SAVE_DIR = "./weights"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 训练超参数（基于iteration）
    MAX_ITERATIONS = 3600  # 最大迭代次数
    VAL_INTERVAL = 90      # 验证间隔
    
    # 实际送入 GPU 的 Batch Size = LOAD_BATCH_SIZE * NUM_SAMPLES
    LOAD_BATCH_SIZE = 1     # 每次从磁盘/缓存读取多少个 Volume (降低 IO 压力)
    NUM_SAMPLES = 32         # 每个 Volume 切多少个 Patch (提高数据利用率)
    
    LR = 1e-4               # 学习率
    ROI_SIZE = (96, 96, 96) # Patch 大小
    
    # 显存/内存优化配置
    NUM_WORKERS = 3
    CACHE_RATE = 1          # 数据缓存比例（1=全部缓存）
    
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
        batch_size=LOAD_BATCH_SIZE, 
        roi_size=ROI_SIZE, 
        num_samples=NUM_SAMPLES, # [新增] 启用多样本采样
        is_train=True, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
    )
    
    val_loader = get_basic_loader(
        data_list=val_files,
        batch_size=1,
        roi_size=ROI_SIZE, 
        is_train=False, 
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
    )

    # ================= 3. 模型与优化器 =================
    model = UNet3D(in_channels=1, out_channels=2).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 4. 训练循环 =================
    best_metric = -1
    best_metric_iteration = -1
    iteration = 0
    epoch_loss = 0
    step_in_epoch = 0
    
    print(f"\n{'='*20} 开始训练 (基于Iteration) {'='*20}")
    print(f"最大迭代次数: {MAX_ITERATIONS}, 验证间隔: {VAL_INTERVAL} iterations")
    
    model.train()
    train_iter = iter(train_loader)
    start_time = time.time() # 记录总开始时间
    loop_start_time = time.time() # 记录循环开始时间
    
    while iteration < MAX_ITERATIONS:
        # 获取下一个batch，如果数据用完则重新开始
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)
        
        iteration += 1
        step_in_epoch += 1
        
        inputs, labels_batch = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_function(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # 打印训练进度
        current_loss = loss.item()
        
        # 计算单次迭代时间
        current_time = time.time()
        iter_time = current_time - loop_start_time
        loop_start_time = current_time # 重置时间起点
        
        print(f"Iteration {iteration}/{MAX_ITERATIONS} | Time: {iter_time:.4f}s | Loss: {current_loss:.4f}")

        # --- Validation ---
        if iteration % VAL_INTERVAL == 0:
            # [优化] 验证前清理显存，为验证阶段腾出空间
            torch.cuda.empty_cache()
            
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

                print(f"Validation at Iter {iteration} | Val Dice: {metric:.4f}", end="")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_iteration = iteration
                    save_path = os.path.join(MODEL_SAVE_DIR, "best_unet3D_model.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f" -> 🔥 New Best! ({best_metric:.4f})")
                else:
                    print("")

            # [优化] 验证后清理显存，释放验证阶段的大量占用，为接下来的训练腾出空间
            torch.cuda.empty_cache()
            
            model.train()
            # 重置统计
            epoch_loss = 0
            step_in_epoch = 0

    total_time = time.time() - start_time
    print(f"\n训练结束。总用时: {total_time:.1f}s")
    print(f"最佳模型 Dice: {best_metric:.4f} 于 Iteration {best_metric_iteration}")

if __name__ == "__main__":
    try:
        train_baseline()
    except Exception as e:
        print(f"❌ 训练发生错误: {e}")
        import traceback
        traceback.print_exc()