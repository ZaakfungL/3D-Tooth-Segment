import os
import sys
import glob
import torch
import numpy as np
import time
import warnings
import json

# 过滤不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="monai.inferers.utils")

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from monai.config import print_config
from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.dataloaders.combo_loader import NASComboDataLoader
from src.models.dints import DiNTSWrapper
from src.ssl.tmo import TMOAdamW
from src.ssl.utils import update_ema_variables, ConsistencyLoss, get_current_consistency_weight

# --- 配置 (目前为硬编码，用于验证) ---
DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
LOG_DIR = "./results/dints_tmo_search"
MAX_EPOCHS = 2
BATCH_SIZE = 2
LR_WEIGHTS = 0.025
LR_ARCH = 0.003
VAL_FREQ = 1
ARCH_START_EPOCH = 0
EMA_ALPHA = 0.99
CONSISTENCY = 10.0
CONSISTENCY_RAMPUP = 50.0
ROI_SIZE = (64, 64, 64) # 根据用户要求更新为 64

def search_tmo():
    # 0. 设置
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(0)
    
    print(f"🚀 开始 DiNTS-TMO 搜索 | 设备: {device}")

    # 1. 模型初始化
    print("初始化模型...")
    # Student: 需要梯度
    student = DiNTSWrapper(
        in_channels=1, 
        out_channels=2, 
        num_blocks=6, 
        num_depths=3
    ).to(device)
    
    # Teacher: 不需要梯度，使用 EMA 权重
    teacher = DiNTSWrapper(
        in_channels=1, 
        out_channels=2, 
        num_blocks=6, 
        num_depths=3
    ).to(device)
    
    # 如果需要，显式设置 dints_space 的设备
    if hasattr(student, "dints_space"):
        student.dints_space.device = device
    if hasattr(teacher, "dints_space"):
        teacher.dints_space.device = device
    
    # 分离 Teacher 参数
    for p in teacher.parameters():
        p.detach_()

    # 2. 数据加载器
    print("初始化 NASComboDataLoader (四路数据流)...")
    combo_loader = NASComboDataLoader(
        data_dir=DATA_DIR,
        batch_size_l=BATCH_SIZE,
        batch_size_u=BATCH_SIZE,
        roi_size=ROI_SIZE,
        limit=4 # 验证模式下限制数据量为 4
    )
    
    # 验证加载器 (暂时复用 basic_loader 或创建新的)
    # 通常应该使用 basic loader 进行验证。
    # 复用 dints_search.py 的验证逻辑
    from src.dataloaders.basic_loader import get_basic_loader
    
    # 简单的验证集创建 (为了本次测试，暂时本地拆分)
    # 实际上 NASCombo 应该只包含训练数据。我们在本地扫描并拆分验证集。
    # 使用有标签数据拆分出一小部分作为伪验证集。
    images_l = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    labels_l = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    dicts_l = [{"image": i, "label": l} for i, l in zip(images_l, labels_l)]
    
    # 取最后 20% 作为伪验证集
    val_split_idx = int(len(dicts_l) * 0.8)
    val_dicts = dicts_l[val_split_idx:]
    if len(val_dicts) == 0: val_dicts = dicts_l # 回退策略
    
    val_loader = get_basic_loader(
        data_list=val_dicts[:2], # 限制验证集大小以加快 dry run
        batch_size=1,
        roi_size=ROI_SIZE,
        is_train=False,
        num_workers=0
    )

    # 3. 优化器初始化
    print("初始化优化器 (TMO)...")
    optimizer_w = TMOAdamW(student.weight_parameters(), lr=LR_WEIGHTS, weight_decay=1e-4)
    optimizer_a = TMOAdamW(student.arch_parameters(), lr=LR_ARCH, weight_decay=1e-3)

    # 4. 损失函数
    loss_dice_ce = DiceCELoss(to_onehot_y=True, softmax=True, batch=True)
    loss_consistency = ConsistencyLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # --- 训练循环 ---
    print(f"\n{'='*20} 开始搜索循环 ({MAX_EPOCHS} epochs) {'='*20}")
    global_step = 0
    best_metric = -1
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        student.train()
        teacher.train() 
        
        loss_w_l_sum = 0
        loss_w_u_sum = 0
        loss_a_l_sum = 0
        loss_a_u_sum = 0
        step = 0
        
        cons_weight = get_current_consistency_weight(epoch, MAX_EPOCHS, CONSISTENCY, CONSISTENCY_RAMPUP)
        
        for batch_data in combo_loader:
            step += 1
            global_step += 1
            
            l_w_imgs, l_w_lbls = batch_data['l_w']['image'].to(device), batch_data['l_w']['label'].to(device)
            u_w_imgs = batch_data['u_w']['image'].to(device)
            l_a_imgs, l_a_lbls = batch_data['l_a']['image'].to(device), batch_data['l_a']['label'].to(device)
            u_a_imgs = batch_data['u_a']['image'].to(device)

            # --- 阶段 A: 优化架构参数 (Alpha) ---
            if epoch >= ARCH_START_EPOCH:
                # A1. 有标签步骤 (Labeled Step)
                optimizer_a.zero_grad()
                outputs_l_a = student(l_a_imgs)
                loss_a_l = loss_dice_ce(outputs_l_a, l_a_lbls)
                
                # 熵损失 (可选，模仿 dints_search 的行为)
                probs_children, _ = student.dints_space.get_prob_a(child=True)
                entropy_loss = student.dints_space.get_topology_entropy(probs_children)
                loss_a_l_total = loss_a_l + 0.001 * entropy_loss
                
                loss_a_l_total.backward()
                optimizer_a.step_labeled()
                loss_a_l_sum += loss_a_l.item()

                # A2. 无标签步骤 (Unlabeled Step)
                optimizer_a.zero_grad()
                
                # 同步 Teacher 架构
                with torch.no_grad():
                    teacher.dints_space.log_alpha_a.copy_(student.dints_space.log_alpha_a)
                    teacher.dints_space.log_alpha_c.copy_(student.dints_space.log_alpha_c)
                
                outputs_u_a = student(u_a_imgs)
                with torch.no_grad():
                    teacher_u_a = teacher(u_a_imgs)
                    teacher_u_a = torch.softmax(teacher_u_a, dim=1)
                
                student_u_a_soft = torch.softmax(outputs_u_a, dim=1)
                loss_a_u = loss_consistency(student_u_a_soft, teacher_u_a) * cons_weight
                
                loss_a_u.backward()
                optimizer_a.step_unlabeled()
                loss_a_u_sum += loss_a_u.item()
            
            # --- 阶段 B: 优化权重参数 (Weights) ---
            # B1. 有标签步骤 (Labeled Step)
            optimizer_w.zero_grad()
            outputs_l_w = student(l_w_imgs)
            loss_w_l = loss_dice_ce(outputs_l_w, l_w_lbls)
            
            loss_w_l.backward()
            optimizer_w.step_labeled()
            loss_w_l_sum += loss_w_l.item()
            
            # B2. 无标签步骤 (Unlabeled Step)
            optimizer_w.zero_grad()
            outputs_u_w = student(u_w_imgs)
            with torch.no_grad():
                teacher_u_w = teacher(u_w_imgs)
                teacher_u_w = torch.softmax(teacher_u_w, dim=1)
            
            student_u_w_soft = torch.softmax(outputs_u_w, dim=1)
            loss_w_u = loss_consistency(student_u_w_soft, teacher_u_w) * cons_weight
            
            loss_w_u.backward()
            optimizer_w.step_unlabeled()
            loss_w_u_sum += loss_w_u.item()

            # --- 阶段 C: 维护 ---
            update_ema_variables(student, teacher, EMA_ALPHA, global_step)

        # Epoch 结束日志
        epoch_time = time.time() - epoch_start
        print(f"Ep {epoch+1}/{MAX_EPOCHS} | Time: {epoch_time:.1f}s | "
              f"L_W(L): {loss_w_l_sum/max(step,1):.4f} L_W(U): {loss_w_u_sum/max(step,1):.4f} | "
              f"L_A(L): {loss_a_l_sum/max(step,1):.4f} L_A(U): {loss_a_u_sum/max(step,1):.4f}", end="")

        # 验证
        if (epoch + 1) % VAL_FREQ == 0:
            teacher.eval() # 使用 Teacher 进行验证
            with torch.no_grad():
                for val_data in val_loader:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    val_pred = sliding_window_inference(val_in, ROI_SIZE, 4, teacher)
                    val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_pred)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_pred, y=val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f" | Val Dice: {metric:.4f}", end="")
                
                if metric > best_metric:
                    best_metric = metric
                    # 保存最佳结果
                    topology = teacher.get_topology()
                    arch_json = {"arch_code_a": topology[1].tolist(), "arch_code_c": topology[2].tolist()}
                    with open(os.path.join(LOG_DIR, "best_arch.json"), "w") as f:
                        json.dump(arch_json, f)
                    torch.save(teacher.state_dict(), os.path.join(LOG_DIR, "model_best.pth"))
                    print(f" -> 🔥 New Best!", end="")
        
        print("")

    print(f"\n搜索结束。最佳 Dice: {best_metric:.4f}")

if __name__ == "__main__":
    try:
        search_tmo()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
