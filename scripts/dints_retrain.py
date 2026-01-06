import sys
import os
import glob
import torch
import json
import warnings
import numpy as np
import time
import argparse

# 忽略 monai 的一些警告
warnings.filterwarnings("ignore", category=UserWarning, module="monai.inferers.utils")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch, partition_dataset
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

# 导入 MONAI DiNTS 离散架构模块
from monai.networks.nets import TopologyInstance, DiNTS

from src.dataloaders.basic_loader import get_basic_loader
from src.utils.config import load_config, get_config_argument_parser

def retrain_from_arch(config):
    # ================= 配置区域 (Fail Fast) =================
    seed = config.get("seed", 2025)
    gpu_id = str(config["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU: {gpu_id}")

    data_dir = config["data_dir"]
    model_save_dir = config["model_save_dir"].format(seed=seed)

    os.makedirs(model_save_dir, exist_ok=True)

    # 1. 查找最佳架构文件
    arch_file = config["arch_file_path"].format(seed=seed)
    if not os.path.exists(arch_file):
        raise FileNotFoundError(f"未找到架构文件: {arch_file}。请在配置中指定正确的 arch_file_path。")

    print(f"载入架构文件: {arch_file}")
    with open(arch_file, "r") as f:
        arch_code = json.load(f)

    # ============ Retrain 阶段参数 ============
    max_iterations = config["max_iterations"]
    val_interval = config["val_interval"]

    batch_size = config["batch_size"]
    num_samples = config["num_samples"]
    roi_size = tuple(config["roi_size"])

    lr_init = config["lr_init"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]

    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    # 验证参数
    val_sw_batch_size = config.get("val_sw_batch_size", 4)
    val_overlap = config.get("val_overlap", 0.5)

    # ================= 1. 数据准备 =================
    set_determinism(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 DiNTS Retrain | Seed: {seed} | 设备: {device}")
    print(f"📂 数据集路径: {data_dir}")

    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    if not images:
        raise ValueError(f"未在 {data_dir} 找到数据！请检查路径。")

    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]

    # 划分训练/验证 (8:2)
    train_files, val_files = partition_dataset(
        data=data_dicts, ratios=[0.8, 0.2], shuffle=True, seed=seed
    )

    print(f"  - 总数据: {len(data_dicts)}")
    print(f"  - 训练集: {len(train_files)}")
    print(f"  - 验证集: {len(val_files)}")

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

    val_loader = get_basic_loader(
        data_list=val_files,
        batch_size=1,
        roi_size=roi_size,
        num_samples=1,
        is_train=False,
        num_workers=num_workers,
        cache_rate=cache_rate,
        shuffle=True
    )

    # ================= 2. 模型定义 (离散架构重构) =================
    # 解析架构编码
    if "arch_code_a_max" in arch_code:
        print("使用 'arch_code_a_max' 作为架构编码")
        arch_code_a = np.array(arch_code["arch_code_a_max"])
    else:
        raise ValueError(f"架构文件 {arch_file} 中未找到 'arch_code_a_max'。请确保使用正确的架构文件。")

    arch_code_c = np.array(arch_code["arch_code_c"])
    arch_code_list = [arch_code_a, arch_code_c]

    # 使用 TopologyInstance 创建离散拓扑 (只保留被选中的操作)
    dints_space = TopologyInstance(
        arch_code=arch_code_list,
        channel_mul=1.0,
        num_blocks=12,
        num_depths=3,
        spatial_dims=3,
        use_downsample=True,
        device=device
    )

    # 创建完整的 DiNTS 网络
    model = DiNTS(
        dints_space=dints_space,
        in_channels=1,
        num_classes=2,
        use_downsample=True,
        spatial_dims=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 优化器：优化所有参数
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr_init,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # 学习率调度：Poly Decay (基于 Iteration)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=max_iterations, power=0.9
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_metric = -1
    best_iter = -1

    # ================= 3. 训练循环 (Iteration Based) =================
    print(f"\n{'='*20} 开始训练 (Steps: {max_iterations}) {'='*20}")

    global_step = 0
    train_iter = iter(train_loader)
    loop_start_time = time.time()

    model.train()

    while global_step < max_iterations:
        global_step += 1

        # 获取数据 (无限循环)
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)

        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()

        # 日志打印
        if global_step % 10 == 0:
            current_time = time.time()
            step_time = current_time - loop_start_time
            loop_start_time = current_time
            print(f"Step {global_step}/{max_iterations} | Time: {step_time:.2f}s | Loss: {current_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # ================= 4. 验证 (遍历整个验证集) =================
        if global_step % val_interval == 0:
            # 释放梯度显存，保留模型权重和优化器状态
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            val_start_time = time.time()

            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)

                    val_pred = sliding_window_inference(
                        val_in, roi_size,
                        sw_batch_size=val_sw_batch_size,
                        predictor=model,
                        overlap=val_overlap
                    )

                    val_pred = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_pred)]
                    val_lbl = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_pred, y=val_lbl)

                    del val_data, val_in, val_lbl, val_pred

                # 聚合所有验证样本的平均Dice
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                val_time = time.time() - val_start_time

                print(f"Validation at Step {global_step} | Val Dice: {metric:.4f} | Val Time: {val_time:.2f}s", end="")

                if metric > best_metric:
                    best_metric = metric
                    best_iter = global_step
                    save_path = os.path.join(model_save_dir, "dints_retrain_best.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f" -> New Best! Model saved to {save_path}")
                else:
                    print("")

            # 恢复训练状态
            torch.cuda.empty_cache()
            model.train()

    print(f"\n训练结束。最佳模型 Dice: {best_metric:.4f} (at Step {best_iter})")

if __name__ == "__main__":
    parser = get_config_argument_parser(description="DiNTS Retrain Script")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    args = parser.parse_args()

    # 加载配置
    default_config_path = os.path.join(project_root, "configs", "dints_retrain.yaml")
    config_path = args.config if args.config else default_config_path

    config = load_config(config_path, default_config=None)

    if not config:
        print(f"❌ 错误: 无法加载配置文件 {config_path}，或文件为空！")
        sys.exit(1)

    if args.seed != 2025:
        config["seed"] = args.seed
    elif "seed" not in config:
        config["seed"] = 2025

    try:
        retrain_from_arch(config)
    except Exception as e:
        print(f"❌ Retrain 失败: {e}")
        import traceback
        traceback.print_exc()
