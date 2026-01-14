"""
DiNTS-TMO 重训练脚本

本脚本使用 TMO (Trusted Momentum Optimization) 半监督框架，
对 DiNTS 搜索阶段找到的最优架构进行重训练。

TMO 核心机制:
  - step_labeled(): 使用有标签梯度更新动量，建立"可信方向" (Trusted Direction)
  - step_unlabeled(): 检查无标签梯度与可信方向的对齐度，进行门控筛选后更新参数

训练流程 (每个 step):
  1. 有标签数据: 计算 DiceCE 损失 -> step_labeled() 建立可信方向
  2. 无标签数据: 计算一致性损失 -> step_unlabeled() 门控筛选 + 参数更新
  3. Teacher 模型更新: EMA 同步权重

运行命令:
    nohup python -u scripts/dints_tmo_retrain.py > results/dints_tmo/retrain/seed2025/retrain.log 2>&1 &
    nohup python -u scripts/dints_tmo_retrain.py --seed 2026 > results/dints_tmo/retrain/seed2026/retrain.log 2>&1 &
"""

import os
import sys
import glob
import time
import json
import warnings

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="monai.inferers.utils")

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch, partition_dataset
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.networks.nets import TopologyInstance, DiNTS

from src.dataloaders.basic_loader import get_basic_loader
from src.ssl.tmo import TMOAdamW
from src.ssl.utils import update_ema_variables, ConsistencyLoss, get_current_consistency_weight
from src.utils.config import load_config, get_config_argument_parser


def cycle(iterable):
    """将有限迭代器转为无限循环迭代器"""
    while True:
        for x in iterable:
            yield x


def retrain_tmo(config):
    """
    DiNTS-TMO 重训练主函数

    Args:
        config: 配置字典，包含数据路径、训练参数、优化器设置等
    """
    # =====================================================================
    # 第一部分: 配置解析
    # =====================================================================
    seed = config.get("seed", 2025)
    gpu_id = str(config.get("gpu_id", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU: {gpu_id}")

    # 路径配置
    data_dir = config["data_dir"]
    weight_dir = config["weight_dir"].format(seed=seed)

    os.makedirs(weight_dir, exist_ok=True)

    # 架构文件
    arch_file = config["arch_file_path"].format(seed=seed)
    if not os.path.exists(arch_file):
        raise FileNotFoundError(f"未找到架构文件: {arch_file}。请先运行搜索阶段。")

    print(f"载入架构文件: {arch_file}")
    with open(arch_file, "r") as f:
        arch_code = json.load(f)

    # 训练参数
    roi_size = tuple(config["roi_size"])
    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    max_iterations = config["max_iterations"]
    val_interval = config["val_interval"]
    batch_size = config["batch_size"]
    num_samples = config["num_samples"]
    unlabeled_ratio = config.get("unlabeled_ratio", 1.0)

    # 验证参数
    val_sw_batch_size = config.get("val_sw_batch_size", 4)
    val_overlap = config.get("val_overlap", 0.5)

    # 优化器参数
    lr_init = config["lr_init"]
    weight_decay = config["weight_decay"]

    # TMO 参数
    ema_alpha = config.get("ema_alpha", 0.99)
    consistency = config.get("consistency", 10.0)
    consistency_rampup_steps = config.get("consistency_rampup_steps", 1000)

    # =====================================================================
    # 第二部分: 数据准备
    # =====================================================================
    set_determinism(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始 DiNTS-TMO Retrain | Seed: {seed} | 设备: {device}")
    print(f"数据集路径: {data_dir}")

    # 扫描有标签数据
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    if not images:
        raise ValueError(f"未在 {data_dir} 找到数据!")

    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]

    # 划分训练/验证 (8:2)
    train_files, val_files = partition_dataset(
        data=data_dicts, ratios=[0.8, 0.2], shuffle=True, seed=seed
    )

    print(f"  - 总数据: {len(data_dicts)}")
    print(f"  - 训练集: {len(train_files)}")
    print(f"  - 验证集: {len(val_files)}")

    # 扫描无标签数据
    images_u = sorted(glob.glob(os.path.join(data_dir, "imagesUnlabeled", "*.nii.gz")))
    if not images_u:
        raise ValueError("TMO 模式需要无标签数据! 请确保 imagesUnlabeled 目录不为空。")

    unlabeled_files = [{"image": i, "label": i} for i in images_u]
    max_unlabeled = int(len(train_files) * unlabeled_ratio)
    if len(unlabeled_files) > max_unlabeled:
        import random
        random.seed(seed)
        unlabeled_files = random.sample(unlabeled_files, max_unlabeled)
    print(f"  - 无标签数据: {len(unlabeled_files)} 例")

    # 创建数据加载器
    train_loader = get_basic_loader(
        data_list=train_files,
        batch_size=batch_size,
        roi_size=roi_size,
        num_samples=num_samples,
        is_train=True,
        num_workers=num_workers,
        cache_rate=cache_rate,
        shuffle=True
    )

    unlabeled_loader = get_basic_loader(
        data_list=unlabeled_files,
        batch_size=batch_size,
        roi_size=roi_size,
        num_samples=num_samples,
        is_train=True,
        num_workers=num_workers,
        cache_rate=cache_rate,
        shuffle=True
    )

    val_loader = get_basic_loader(
        data_list=val_files,
        batch_size=1,
        roi_size=roi_size,
        num_samples=1,
        is_train=False,
        num_workers=num_workers,
        cache_rate=cache_rate,
        shuffle=False
    )

    # =====================================================================
    # 第三部分: 模型定义 (离散架构重构 + Student-Teacher)
    # =====================================================================
    # 解析架构编码 (优先使用 arch_code_a_max)
    if "arch_code_a_max" in arch_code:
        print("使用 'arch_code_a_max' 作为架构编码")
        arch_code_a = np.array(arch_code["arch_code_a_max"], dtype=np.int64)
    elif "arch_code_a" in arch_code:
        print("警告: 未找到 'arch_code_a_max'，退级使用 'arch_code_a'")
        arch_code_a = np.array(arch_code["arch_code_a"], dtype=np.float64)
    else:
        raise ValueError(f"架构文件中未找到 'arch_code_a_max' 或 'arch_code_a'，请检查架构文件格式。")

    arch_code_c = np.array(arch_code["arch_code_c"])
    arch_code_list = [arch_code_a, arch_code_c]

    # 创建 Student 模型
    dints_space_student = TopologyInstance(
        arch_code=arch_code_list,
        channel_mul=1.0,
        num_blocks=12,
        num_depths=3,
        spatial_dims=3,
        use_downsample=True,
        device=device
    )

    student = DiNTS(
        dints_space=dints_space_student,
        in_channels=1,
        num_classes=2,
        use_downsample=True,
        spatial_dims=3,
    ).to(device)

    # 创建 Teacher 模型 (相同架构)
    dints_space_teacher = TopologyInstance(
        arch_code=arch_code_list,
        channel_mul=1.0,
        num_blocks=12,
        num_depths=3,
        spatial_dims=3,
        use_downsample=True,
        device=device
    )

    teacher = DiNTS(
        dints_space=dints_space_teacher,
        in_channels=1,
        num_classes=2,
        use_downsample=True,
        spatial_dims=3,
    ).to(device)

    # 初始化 Teacher 权重与 Student 相同
    teacher.load_state_dict(student.state_dict())

    # Teacher 不参与梯度计算
    for p in teacher.parameters():
        p.detach_()

    total_params = sum(p.numel() for p in student.parameters())
    print(f"模型总参数量: {total_params:,}")

    # =====================================================================
    # 第四部分: 优化器与损失函数
    # =====================================================================
    optimizer = TMOAdamW(
        student.parameters(),
        lr=lr_init,
        weight_decay=weight_decay
    )

    # 使用 lambda 方式实现多项式衰减
    lr_lambda = lambda step: max(0.0, (1 - step / max_iterations) ** 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_consistency = ConsistencyLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # =====================================================================
    # 第五部分: 训练循环
    # =====================================================================
    print(f"\n{'='*20} 开始训练 (Steps: {max_iterations}) {'='*20}")

    best_metric = -1
    best_iter = -1
    global_step = 0

    train_iter = cycle(train_loader)
    unlabeled_iter = cycle(unlabeled_loader)

    loop_start_time = time.time()

    while global_step < max_iterations:
        global_step += 1
        student.train()
        teacher.train()

        # 一致性权重
        cons_weight = get_current_consistency_weight(
            global_step, max_iterations, consistency, consistency_rampup_steps
        )

        # -----------------------------------------------------------------
        # 阶段 A: 有标签数据训练 (TMO step_labeled)
        # -----------------------------------------------------------------
        optimizer.zero_grad()
        batch_l = next(train_iter)
        inputs_l = batch_l["image"].to(device)
        labels_l = batch_l["label"].to(device)

        outputs_l = student(inputs_l)
        loss_l = loss_dice_ce(outputs_l, labels_l)
        loss_l.backward()
        optimizer.step_labeled()  # TMO 第一步: 建立可信方向
        del inputs_l, labels_l, outputs_l

        # -----------------------------------------------------------------
        # 阶段 B: 无标签数据训练 (TMO step_unlabeled)
        # -----------------------------------------------------------------
        optimizer.zero_grad()  # 必须清零梯度
        batch_u = next(unlabeled_iter)
        inputs_u = batch_u["image"].to(device)

        with torch.no_grad():
            teacher_u = teacher(inputs_u)
            teacher_soft = torch.softmax(teacher_u, dim=1)
        del teacher_u

        outputs_u = student(inputs_u)
        student_u_soft = torch.softmax(outputs_u, dim=1)
        loss_u = loss_consistency(student_u_soft, teacher_soft) * cons_weight
        loss_u.backward()
        optimizer.step_unlabeled()  # TMO 第二步: 门控筛选 + 参数更新
        del inputs_u, outputs_u, teacher_soft, student_u_soft

        scheduler.step()

        # -----------------------------------------------------------------
        # 阶段 C: Teacher 模型更新 (EMA)
        # -----------------------------------------------------------------
        update_ema_variables(student, teacher, ema_alpha, global_step)

        # 日志打印
        if global_step % 10 == 0:
            current_time = time.time()
            step_time = current_time - loop_start_time
            loop_start_time = current_time
            print(f"Step {global_step}/{max_iterations} | Time: {step_time:.2f}s | "
                  f"Loss L: {loss_l.item():.4f} | Loss U: {loss_u.item():.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # -----------------------------------------------------------------
        # 验证
        # -----------------------------------------------------------------
        if global_step % val_interval == 0:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()  # 等待所有 GPU 操作完成
            torch.cuda.empty_cache()
            val_start_time = time.time()

            teacher.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_in = val_data["image"].to(device)
                    val_lbl = val_data["label"].to(device)

                    val_pred = sliding_window_inference(
                        val_in, roi_size,
                        sw_batch_size=val_sw_batch_size,
                        predictor=teacher,
                        overlap=val_overlap
                    )

                    val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i)
                                for i in decollate_batch(val_pred)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i)
                               for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_pred, y=val_lbl)

                    del val_data, val_in, val_lbl, val_pred

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                val_time = time.time() - val_start_time

                print(f"Validation at Step {global_step} | Val Dice: {metric:.4f} | "
                      f"Val Time: {val_time:.2f}s", end="")

                if metric > best_metric:
                    best_metric = metric
                    best_iter = global_step
                    save_path = os.path.join(weight_dir, "best.pth")
                    torch.save(teacher.state_dict(), save_path)
                    print(f" -> New Best! Model saved to {save_path}")
                else:
                    print("")

            torch.cuda.empty_cache()
            student.train()
            teacher.train()

    print(f"\n训练结束。最佳模型 Dice: {best_metric:.4f} (at Step {best_iter})")


if __name__ == "__main__":
    parser = get_config_argument_parser(description="DiNTS TMO Retrain Script")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    args = parser.parse_args()

    # 加载配置
    default_config_path = os.path.join(project_root, "configs", "dints_tmo_retrain.yaml")
    config_path = args.config if args.config else default_config_path

    config = load_config(config_path, default_config=None)

    if not config:
        print(f"错误: 无法加载配置文件 {config_path}，或文件为空!")
        sys.exit(1)

    if args.seed != 2025:
        config["seed"] = args.seed
    elif "seed" not in config:
        config["seed"] = 2025

    try:
        retrain_tmo(config)
    except Exception as e:
        print(f"Retrain 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
