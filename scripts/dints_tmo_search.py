import os
import sys
import glob
import torch
import numpy as np
import time
import warnings
import json
import argparse

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
from src.utils.config import load_config, get_config_argument_parser

def search_tmo(config):
    # ================= 配置读取 (Fail Fast) =================
    # 基础配置
    data_dir = config["data_dir"]
    log_dir = config["log_dir"]
    roi_size = tuple(config["roi_size"])
    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    # 训练参数
    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    val_freq = config["val_freq"]
    arch_start_epoch = config["arch_start_epoch"]
    unlabeled_ratio = config["unlabeled_ratio"]

    # 优化器与损失
    lr_weights = config["lr_weights"]
    lr_arch = config["lr_arch"]
    ema_alpha = config["ema_alpha"]
    consistency = config["consistency"]
    consistency_rampup = config["consistency_rampup"]

    seed = config.get("seed", 2025)

    # 0. 设置
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(0) # TMO script used 0 in original code, keeping it or using seed? Original used 0.
    # Actually, let's use the seed from config if provided, or default to 0 if that's what was intended.
    # Original code had `set_determinism(0)` hardcoded. Let's use `seed`.
    set_determinism(seed)

    print(f"🚀 开始 DiNTS-TMO 搜索 | 设备: {device} | Seed: {seed}")
    print(f"📂 数据集: {data_dir}")

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
    print("初始化数据加载器...")

    # ===== 先划分 Train/Val，避免数据泄漏 =====
    from monai.data import partition_dataset
    from src.dataloaders.basic_loader import get_basic_loader

    # 扫描有标签数据
    images_l = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels_l = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    if not images_l:
         raise ValueError(f"未在 {data_dir} 找到有标签数据！")

    all_labeled = [{"image": i, "label": l} for i, l in zip(images_l, labels_l)]

    # 划分 Train/Val (80%/20%)，保证互斥
    train_labeled, val_labeled = partition_dataset(
        data=all_labeled, ratios=[0.8, 0.2], shuffle=True, seed=seed
    )
    print(f"📊 Train/Val 划分: Train {len(train_labeled)} 例 | Val {len(val_labeled)} 例")

    # 扫描无标签数据 (不需要划分，全部用于训练)
    images_u = sorted(glob.glob(os.path.join(data_dir, "imagesUnlabeled", "*.nii.gz")))

    if not images_u:
        print("⚠️ 警告: 未找到无标签数据 (imagesUnlabeled)，将仅使用有标签数据。")
        all_unlabeled = []
    else:
        all_unlabeled = [{"image": i, "label": i} for i in images_u]  # label 占位，不使用

        # ===== 控制有标签/无标签比例 =====
        # 按 1:1 比例截取无标签数据，避免加载过多数据浪费 RAM
        # unlabeled_ratio (1.0 = 1:1)
        max_unlabeled = int(len(train_labeled) * unlabeled_ratio)
        if len(all_unlabeled) > max_unlabeled:
            # 随机采样，保证多样性
            import random
            random.seed(seed)
            all_unlabeled = random.sample(all_unlabeled, max_unlabeled)
        print(f"📊 无标签数据: {len(all_unlabeled)} 例 (比例 1:{unlabeled_ratio})")

    # ===== 创建 NASComboDataLoader (只用训练数据) =====
    combo_loader = NASComboDataLoader(
        labeled_list=train_labeled,      # 传入划分好的有标签训练集
        unlabeled_list=all_unlabeled,    # 传入全部无标签数据
        batch_size_l=batch_size,
        batch_size_u=batch_size,
        roi_size=roi_size,
        num_workers=num_workers,
        cache_rate=cache_rate,
        # limit=4  # 调试时取消注释
    )

    # ===== 创建验证加载器 =====
    val_loader = get_basic_loader(
        data_list=val_labeled,
        batch_size=1,
        roi_size=roi_size,
        is_train=False,
        num_workers=num_workers,
        cache_rate=cache_rate
    )

    # 3. 优化器初始化
    print("初始化优化器 (TMO)...")
    optimizer_w = TMOAdamW(student.weight_parameters(), lr=lr_weights, weight_decay=1e-4)
    optimizer_a = TMOAdamW(student.arch_parameters(), lr=lr_arch, weight_decay=1e-3)

    # 4. 损失函数
    loss_dice_ce = DiceCELoss(to_onehot_y=True, softmax=True, batch=True)
    loss_consistency = ConsistencyLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # --- 训练循环 ---
    print(f"\n{'='*20} 开始搜索循环 ({max_epochs} epochs) {'='*20}")
    global_step = 0
    best_metric = -1

    for epoch in range(max_epochs):
        epoch_start = time.time()
        student.train()
        teacher.train()

        loss_w_l_sum = 0
        loss_w_u_sum = 0
        loss_a_l_sum = 0
        loss_a_u_sum = 0
        step = 0

        cons_weight = get_current_consistency_weight(epoch, max_epochs, consistency, consistency_rampup)

        for batch_data in combo_loader:
            step += 1
            global_step += 1

            l_w_imgs, l_w_lbls = batch_data['l_w']['image'].to(device), batch_data['l_w']['label'].to(device)
            u_w_imgs = batch_data['u_w']['image'].to(device)
            l_a_imgs, l_a_lbls = batch_data['l_a']['image'].to(device), batch_data['l_a']['label'].to(device)
            u_a_imgs = batch_data['u_a']['image'].to(device)

            # --- 阶段 A: 优化架构参数 (Alpha) ---
            if epoch >= arch_start_epoch:
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

            # 同步 Teacher 架构到最新状态 (确保 Teacher 使用最新的架构参数)
            with torch.no_grad():
                teacher.dints_space.log_alpha_a.copy_(student.dints_space.log_alpha_a)
                teacher.dints_space.log_alpha_c.copy_(student.dints_space.log_alpha_c)

            outputs_u_w = student(u_w_imgs)
            with torch.no_grad():
                teacher_u_w = teacher(u_w_imgs)
                teacher_u_w = torch.softmax(teacher_u_w, dim=1)

            student_u_w_soft = torch.softmax(outputs_u_w, dim=1)
            loss_w_u = loss_consistency(student_u_w_soft, teacher_u_w) * cons_weight

            loss_w_u.backward()
            optimizer_w.step_unlabeled()
            loss_w_u_sum += loss_w_u.item()

            # --- 阶段 C: 维护 Teacher 模型 ---
            # C1. EMA 更新权重参数
            update_ema_variables(student, teacher, ema_alpha, global_step)

            # C2. 同步架构参数 (有两种策略，根据 Algorithm 3 第10行)
            # 策略 A: 直接复用 Student 的最新架构 (推荐，更简单)
            # 策略 B: 对架构参数也做 EMA (对应图片算法)
            # 这里采用策略 A，因为架构参数变化较慢，直接同步更稳定
            with torch.no_grad():
                teacher.dints_space.log_alpha_a.copy_(student.dints_space.log_alpha_a)
                teacher.dints_space.log_alpha_c.copy_(student.dints_space.log_alpha_c)

        # Epoch 结束日志
        epoch_time = time.time() - epoch_start
        print(f"Ep {epoch+1}/{max_epochs} | Time: {epoch_time:.1f}s | "
              f"L_W(L): {loss_w_l_sum/max(step,1):.4f} L_W(U): {loss_w_u_sum/max(step,1):.4f} | "
              f"L_A(L): {loss_a_l_sum/max(step,1):.4f} L_A(U): {loss_a_u_sum/max(step,1):.4f}", end="")

        # 验证
        if (epoch + 1) % val_freq == 0:
            teacher.eval() # 使用 Teacher 进行验证
            with torch.no_grad():
                for val_data in val_loader:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    val_pred = sliding_window_inference(val_in, roi_size, 4, teacher)
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
                    with open(os.path.join(log_dir, "best_arch.json"), "w") as f:
                        json.dump(arch_json, f)
                    torch.save(teacher.state_dict(), os.path.join(log_dir, "model_best.pth"))
                    print(f" -> 🔥 New Best!", end="")

        print("")

    print(f"\n搜索结束。最佳 Dice: {best_metric:.4f}")

if __name__ == "__main__":
    parser = get_config_argument_parser(description="DiNTS TMO Search Script")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    args = parser.parse_args()

    # 加载配置
    default_config_path = os.path.join(project_root, "configs", "dints_tmo_search.yaml")
    config_path = args.config if args.config else default_config_path

    config = load_config(config_path, default_config=None)

    if not config:
        print(f"❌ 错误: 无法加载配置文件 {config_path}，或文件为空！")
        sys.exit(1)

    # 优先级: Args > Config > Default
    if args.seed != 2025:
        config["seed"] = args.seed
    elif "seed" not in config:
        config["seed"] = 2025

    try:
        search_tmo(config)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
