"""
DiNTS-TMO 架构搜索脚本

本脚本实现了基于 TMO (Teacher-guided Multi-Objective) 的 DiNTS 神经架构搜索。
核心思路: 使用 Student-Teacher 框架进行半监督学习，同时搜索最优网络拓扑。

训练流程 (每个 step):
  1. 架构参数优化 (Alpha): 有标签 -> 无标签 (一致性损失)
  2. 权重参数优化 (Weights): 有标签 -> 无标签 (一致性损失)
  3. Teacher 模型更新: EMA 同步权重 + 复制架构参数

运行命令:
    nohup python -u scripts/dints_tmo_search.py > results/dints_tmo/search/seed2025/search.log 2>&1 &
    nohup python -u scripts/dints_tmo_search.py --seed 2026 > results/dints_tmo/search/seed2026/search.log 2>&1 &
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
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.dataloaders.combo_loader import NASComboDataLoader
from src.models.dints import DiNTSWrapper
from src.ssl.tmo import TMOAdamW
from src.ssl.utils import update_ema_variables, ConsistencyLoss, get_current_consistency_weight
from src.utils.config import load_config, get_config_argument_parser


def cycle(iterable):
    """将有限迭代器转为无限循环迭代器"""
    while True:
        for x in iterable:
            yield x

def search_tmo(config):
    """
    DiNTS-TMO 架构搜索主函数

    Args:
        config: 配置字典，包含数据路径、训练参数、优化器设置等
    """
    # =====================================================================
    # 第一部分: 配置解析
    # =====================================================================
    # GPU 设置
    gpu_id = str(config.get("gpu_id", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU: {gpu_id}")

    # 数据与日志
    data_dir = config["data_dir"]
    seed = config.get("seed", 2025)
    result_dir = config["result_dir"].format(seed=seed)
    weight_dir = config["weight_dir"].format(seed=seed)
    roi_size = tuple(config["roi_size"])
    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    # 训练控制
    max_iterations = config["max_iterations"]      # 总训练步数
    batch_size = config["batch_size"]
    eval_interval = config["eval_interval"]        # 验证间隔 (步数)
    arch_search_start_steps = config["arch_search_start_steps"]  # 架构搜索延迟启动
    unlabeled_ratio = config["unlabeled_ratio"]    # 无标签/有标签比例

    # 优化器
    lr_weights = config["lr_weights"]              # 权重学习率
    lr_arch = config["lr_arch"]                    # 架构参数学习率
    ema_alpha = config["ema_alpha"]                # Teacher EMA 衰减系数
    consistency = config["consistency"]            # 一致性损失权重上限
    consistency_rampup_steps = config["consistency_rampup_steps"]  # 一致性权重预热步数

    # 验证参数
    val_sw_batch_size = config.get("val_sw_batch_size", 4)
    val_overlap = config.get("val_overlap", 0.5)

    seed = config.get("seed", 2025)

    # 初始化环境
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed)

    print(f"开始 DiNTS-TMO 搜索 | 设备: {device} | Seed: {seed}")
    print(f"数据集: {data_dir}")

    # =====================================================================
    # 第二部分: 模型初始化 (Student-Teacher 架构)
    # =====================================================================
    # Student: 主模型，参与梯度优化
    # Teacher: 影子模型，权重通过 EMA 从 Student 同步，用于生成伪标签
    print("初始化模型...")
    student = DiNTSWrapper(
        in_channels=1, out_channels=2, num_blocks=12, num_depths=3
    ).to(device)

    teacher = DiNTSWrapper(
        in_channels=1, out_channels=2, num_blocks=12, num_depths=3
    ).to(device)

    # 确保 dints_space 设备一致
    if hasattr(student, "dints_space"):
        student.dints_space.device = device
    if hasattr(teacher, "dints_space"):
        teacher.dints_space.device = device

    total_params = sum(p.numel() for p in student.parameters())
    print(f"模型总参数量: {total_params:,}")

    # Teacher 不参与梯度计算，只通过 EMA 更新
    for p in teacher.parameters():
        p.detach_()

    # =====================================================================
    # 第三部分: 数据加载
    # =====================================================================
    # 数据流设计:
    #   - 有标签数据: 80% 训练 / 20% 验证 (互斥，防止数据泄漏)
    #   - 无标签数据: 全部用于训练
    #   - NASComboDataLoader 返回 4 路数据: l_w, l_a, u_w, u_a
    #     (l=labeled, u=unlabeled, w=weights优化用, a=arch优化用)
    print("初始化数据加载器...")

    from monai.data import partition_dataset
    from src.dataloaders.basic_loader import get_basic_loader

    # 扫描有标签数据
    images_l = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels_l = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    if not images_l:
        raise ValueError(f"未在 {data_dir} 找到有标签数据!")

    all_labeled = [{"image": i, "label": l} for i, l in zip(images_l, labels_l)]

    # Train/Val 划分 (80%/20%)
    train_labeled, val_labeled = partition_dataset(
        data=all_labeled, ratios=[0.8, 0.2], shuffle=True, seed=seed
    )
    print(f"Train/Val 划分: Train {len(train_labeled)} 例 | Val {len(val_labeled)} 例")

    # 扫描无标签数据
    images_u = sorted(glob.glob(os.path.join(data_dir, "imagesUnlabeled", "*.nii.gz")))
    if not images_u:
        print("警告: 未找到无标签数据 (imagesUnlabeled)，将仅使用有标签数据。")
        all_unlabeled = []
    else:
        # label 字段为占位符，不会实际使用
        all_unlabeled = [{"image": i, "label": i} for i in images_u]

        # 按比例控制无标签数据量，避免内存浪费
        max_unlabeled = int(len(train_labeled) * unlabeled_ratio)
        if len(all_unlabeled) > max_unlabeled:
            import random
            random.seed(seed)
            all_unlabeled = random.sample(all_unlabeled, max_unlabeled)
        print(f"无标签数据: {len(all_unlabeled)} 例 (比例 1:{unlabeled_ratio})")

    # 创建训练数据加载器
    combo_loader = NASComboDataLoader(
        labeled_list=train_labeled,
        unlabeled_list=all_unlabeled,
        batch_size_l=batch_size,
        batch_size_u=batch_size,
        roi_size=roi_size,
        num_workers=num_workers,
        cache_rate=cache_rate,
    )

    # 创建验证数据加载器
    val_loader = get_basic_loader(
        data_list=val_labeled,
        batch_size=1,
        roi_size=roi_size,
        is_train=False,
        num_workers=num_workers,
        cache_rate=cache_rate
    )

    # =====================================================================
    # 第四部分: 优化器与损失函数
    # =====================================================================
    # TMOAdamW: 支持分步优化 (step_labeled / step_unlabeled)
    print("初始化优化器 (TMO)...")
    optimizer_w = TMOAdamW(student.weight_parameters(), lr=lr_weights, weight_decay=1e-4)
    optimizer_a = TMOAdamW(student.arch_parameters(), lr=lr_arch, weight_decay=1e-3)

    loss_dice_ce = DiceCELoss(to_onehot_y=True, softmax=True, batch=True)
    loss_consistency = ConsistencyLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # =====================================================================
    # 第五部分: 训练循环
    # =====================================================================
    # 每个 step 的执行流程:
    #   阶段 A: 架构参数优化 (在 arch_search_start_steps 之后启动)
    #     A1. 有标签数据: 计算 DiceCE + 熵正则，更新 alpha
    #     A2. 无标签数据: 计算一致性损失 (Student vs Teacher)，更新 alpha
    #   阶段 B: 权重参数优化
    #     B1. 有标签数据: 计算 DiceCE，更新 weights
    #     B2. 无标签数据: 计算一致性损失，更新 weights
    #   阶段 C: Teacher 维护
    #     C1. EMA 更新 Teacher 权重
    #     C2. 同步 Teacher 架构参数 (直接复制)
    print(f"\n{'='*20} 开始搜索循环 ({max_iterations} steps) {'='*20}")

    global_step = 0
    best_metric = -1

    # 损失累积 (用于日志)
    loss_w_l_sum = 0  # 权重优化-有标签损失
    loss_w_u_sum = 0  # 权重优化-无标签损失
    loss_a_l_sum = 0  # 架构优化-有标签损失
    loss_a_u_sum = 0  # 架构优化-无标签损失

    # 日志间隔: 每遍历完一轮数据集打印一次
    steps_per_round = len(combo_loader)
    log_interval_steps = steps_per_round

    # 无限迭代器，支持跨 epoch 连续训练
    loader_iter = cycle(combo_loader)

    loop_start_time = time.time()

    while global_step < max_iterations:
        global_step += 1
        student.train()
        teacher.train()

        batch_data = next(loader_iter)

        # 一致性权重: 从 0 逐渐增加到 consistency，用于控制无标签损失贡献
        cons_weight = get_current_consistency_weight(
            global_step, max_iterations, consistency, consistency_rampup_steps
        )

        # -----------------------------------------------------------------
        # 阶段 A: 架构参数优化 (延迟启动，等权重初步收敛)
        # -----------------------------------------------------------------
        if global_step >= arch_search_start_steps:
            # A1. 有标签步骤
            optimizer_a.zero_grad()
            l_a_imgs = batch_data['l_a']['image'].to(device)
            l_a_lbls = batch_data['l_a']['label'].to(device)

            outputs_l_a = student(l_a_imgs)
            loss_a_l = loss_dice_ce(outputs_l_a, l_a_lbls)

            # 熵正则: 鼓励架构概率分布更确定
            probs_children, _ = student.dints_space.get_prob_a(child=True)
            entropy_loss = student.dints_space.get_topology_entropy(probs_children)
            loss_a_l_total = loss_a_l + 0.001 * entropy_loss

            loss_a_l_total.backward()
            optimizer_a.step_labeled()
            loss_a_l_sum += loss_a_l.item()

            del l_a_imgs, l_a_lbls, outputs_l_a, loss_a_l, loss_a_l_total

            # A2. 无标签步骤
            optimizer_a.zero_grad()

            # 同步 Teacher 架构参数
            with torch.no_grad():
                teacher.dints_space.log_alpha_a.copy_(student.dints_space.log_alpha_a)
                teacher.dints_space.log_alpha_c.copy_(student.dints_space.log_alpha_c)

            u_a_imgs = batch_data['u_a']['image'].to(device)
            outputs_u_a = student(u_a_imgs)

            with torch.no_grad():
                teacher_u_a = teacher(u_a_imgs)
                teacher_u_a = torch.softmax(teacher_u_a, dim=1)

            student_u_a_soft = torch.softmax(outputs_u_a, dim=1)
            loss_a_u = loss_consistency(student_u_a_soft, teacher_u_a) * cons_weight

            loss_a_u.backward()
            optimizer_a.step_unlabeled()
            loss_a_u_sum += loss_a_u.item()

            del u_a_imgs, outputs_u_a, teacher_u_a, student_u_a_soft, loss_a_u

        # -----------------------------------------------------------------
        # 阶段 B: 权重参数优化
        # -----------------------------------------------------------------
        # B1. 有标签步骤
        optimizer_w.zero_grad()
        l_w_imgs = batch_data['l_w']['image'].to(device)
        l_w_lbls = batch_data['l_w']['label'].to(device)

        outputs_l_w = student(l_w_imgs)
        loss_w_l = loss_dice_ce(outputs_l_w, l_w_lbls)

        loss_w_l.backward()
        optimizer_w.step_labeled()
        loss_w_l_sum += loss_w_l.item()

        del l_w_imgs, l_w_lbls, outputs_l_w, loss_w_l

        # B2. 无标签步骤
        optimizer_w.zero_grad()

        # 同步 Teacher 架构参数 (确保使用最新架构)
        with torch.no_grad():
            teacher.dints_space.log_alpha_a.copy_(student.dints_space.log_alpha_a)
            teacher.dints_space.log_alpha_c.copy_(student.dints_space.log_alpha_c)

        u_w_imgs = batch_data['u_w']['image'].to(device)
        outputs_u_w = student(u_w_imgs)

        with torch.no_grad():
            teacher_u_w = teacher(u_w_imgs)
            teacher_u_w = torch.softmax(teacher_u_w, dim=1)

        student_u_w_soft = torch.softmax(outputs_u_w, dim=1)
        loss_w_u = loss_consistency(student_u_w_soft, teacher_u_w) * cons_weight

        loss_w_u.backward()
        optimizer_w.step_unlabeled()
        loss_w_u_sum += loss_w_u.item()

        del u_w_imgs, outputs_u_w, teacher_u_w, student_u_w_soft, loss_w_u

        # -----------------------------------------------------------------
        # 阶段 C: Teacher 模型维护
        # -----------------------------------------------------------------
        # EMA 更新权重
        update_ema_variables(student, teacher, ema_alpha, global_step)

        # 同步架构参数 (直接复制，不做 EMA)
        with torch.no_grad():
            teacher.dints_space.log_alpha_a.copy_(student.dints_space.log_alpha_a)
            teacher.dints_space.log_alpha_c.copy_(student.dints_space.log_alpha_c)

        # -----------------------------------------------------------------
        # 日志与验证
        # -----------------------------------------------------------------
        if global_step % log_interval_steps == 0:
            current_round = global_step // steps_per_round
            interval_time = time.time() - loop_start_time

            avg_lw_l = loss_w_l_sum / log_interval_steps
            avg_lw_u = loss_w_u_sum / log_interval_steps
            avg_la_l = loss_a_l_sum / log_interval_steps
            avg_la_u = loss_a_u_sum / log_interval_steps

            print(f"Step {global_step}/{max_iterations} (Round {current_round}) | "
                  f"Time: {interval_time:.1f}s | "
                  f"L_W(L): {avg_lw_l:.4f} L_W(U): {avg_lw_u:.4f} | "
                  f"L_A(L): {avg_la_l:.4f} L_A(U): {avg_la_u:.4f}", end="")

            # 重置累积器
            loss_w_l_sum = loss_w_u_sum = loss_a_l_sum = loss_a_u_sum = 0
            loop_start_time = time.time()

            # 验证
            if global_step % eval_interval == 0:
                val_start_time = time.time()
                teacher.eval()

                with torch.no_grad():
                    for val_data in val_loader:
                        val_in = val_data["image"].to(device)
                        val_lbl = val_data["label"].to(device)
                        val_pred = sliding_window_inference(
                            val_in, roi_size, val_sw_batch_size, teacher,
                            overlap=val_overlap
                        )
                        val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i)
                                    for i in decollate_batch(val_pred)]
                        val_lbl = [AsDiscrete(to_onehot=2)(i)
                                   for i in decollate_batch(val_lbl)]
                        dice_metric(y_pred=val_pred, y=val_lbl)

                        # 及时释放显存
                        del val_in, val_lbl, val_pred

                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    val_time = time.time() - val_start_time

                # 验证结束后清理显存
                torch.cuda.empty_cache()

                print(f" | Val Dice: {metric:.4f} | Val Time: {val_time:.2f}s", end="")

                # 保存最佳模型
                if metric > best_metric:
                    best_metric = metric
                    topology = teacher.get_topology()
                    arch_json = {
                        "arch_code_a": topology[1].tolist(),
                        "arch_code_c": topology[2].tolist(),
                        "arch_code_a_max": topology[3].tolist() if len(topology) > 3 else topology[1].tolist()
                    }
                    with open(os.path.join(result_dir, "arch.json"), "w") as f:
                        json.dump(arch_json, f, indent=4)
                    torch.save(teacher.state_dict(),
                               os.path.join(weight_dir, "best.pth"))
                    print(f" -> New Best!", end="")

            print("")  # 换行

    print(f"\n搜索结束。最佳 Dice: {best_metric:.4f}")

if __name__ == "__main__":
    parser = get_config_argument_parser(description="DiNTS TMO Search Script")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()

    default_config_path = os.path.join(project_root, "configs", "dints_tmo_search.yaml")
    config_path = args.config if args.config else default_config_path
    config = load_config(config_path, default_config=None)

    if not config:
        print(f"错误: 无法加载配置文件 {config_path}")
        sys.exit(1)

    # 参数优先级: 命令行 > 配置文件 > 默认值
    if args.seed != 2025:
        config["seed"] = args.seed
    elif "seed" not in config:
        config["seed"] = 2025

    try:
        search_tmo(config)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
