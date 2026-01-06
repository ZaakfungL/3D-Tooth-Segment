import sys
import os
import glob
import torch
import json
import warnings
import time
import argparse

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

# 导入你的模块
from src.models.dints import DiNTSWrapper
from src.dataloaders.basic_loader import get_basic_loader
from src.utils.config import load_config, get_config_argument_parser

def search_baseline(config):
    # ================= 配置读取 (Fail Fast) =================
    # 基础配置
    seed = config.get("seed", 2025)
    gpu_id = str(config["gpu_id"])

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU: {gpu_id}")

    # 路径配置
    data_dir = config["data_dir"]
    arch_save_dir = config["arch_save_dir"].format(seed=seed)

    os.makedirs(arch_save_dir, exist_ok=True)

    # 搜索阶段参数
    warmup_steps = config["warmup_steps"]
    arch_search_start_steps = config["arch_search_start_steps"]
    max_iterations = config["max_iterations"]
    eval_interval = config["eval_interval"]

    # 验证优化参数
    num_val_samples = config.get("num_val_samples", 0)
    val_sw_batch_size = config.get("val_sw_batch_size", 8)
    val_overlap = config.get("val_overlap", 0.5)

    lr_decay_steps = config["lr_decay_steps"]
    lr_decay_factor = config["lr_decay_factor"]

    batch_size = config["batch_size"]
    num_samples = config["num_samples"]
    roi_size = tuple(config["roi_size"])

    # 学习率
    lr_weights_init = config["lr_weights_init"]
    lr_weights_max = config["lr_weights_max"]
    lr_arch = config["lr_arch"]

    # 资源
    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    # ================= 1. 数据准备 =================
    set_determinism(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 DiNTS 搜索 | Seed: {seed} | 设备: {device}")
    print(f"👉 策略: Warmup={warmup_steps} | ArchStart={arch_search_start_steps} | Total={max_iterations}")
    print(f"📂 数据集路径: {data_dir}")

    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    if not images:
        raise ValueError(f"❌ 未在 {data_dir} 找到任何数据，请检查路径配置！")

    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]

    # NAS 数据划分
    train_files_all, val_files = partition_dataset(
        data=data_dicts, ratios=[0.8, 0.2], shuffle=True, seed=seed
    )

    train_files_w, train_files_a = partition_dataset(
        data=train_files_all, ratios=[0.5, 0.5], shuffle=True, seed=seed
    )

    print(f"  - 总数据: {len(data_dicts)}")
    print(f"  - 权重更新集 (Train_W): {len(train_files_w)}")
    print(f"  - 架构更新集 (Train_A): {len(train_files_a)}")
    print(f"  - 验证集 (Val): {len(val_files)}")

    train_loader_w = get_basic_loader(
        data_list=train_files_w,
        batch_size=batch_size,
        roi_size=roi_size,
        num_samples=num_samples,
        is_train=True,
        num_workers=num_workers,
        cache_rate=cache_rate,
        shuffle=True
    )

    train_loader_a = get_basic_loader(
        data_list=train_files_a,
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

    # ================= 2. 模型与双优化器 =================
    model = DiNTSWrapper(
        in_channels=1,
        out_channels=2,
        num_blocks=12,
        num_depths=3,
        channel_mul=1,
        use_downsample=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    if hasattr(model, "dints_space"):
        model.dints_space.device = device

    optimizer_w = torch.optim.SGD(
        model.weight_parameters(),
        lr=lr_weights_init,
        momentum=0.9,
        weight_decay=4e-5
    )

    optimizer_a = torch.optim.Adam(
        model.arch_parameters(),
        lr=lr_arch,
        betas=(0.5, 0.999),
        weight_decay=0
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 3. 搜索循环 (Batch Iteration) =================
    best_metric = -1
    best_metric_step = -1

    print(f"\n{'='*20} 开始搜索 (Steps: {max_iterations}) {'='*20}")

    global_step = 0
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    iter_w = cycle(train_loader_w)
    iter_a = cycle(train_loader_a)

    loop_start_time = time.time()

    while global_step < max_iterations:
        global_step += 1
        model.train()

        # ------------------------------------------------
        # 学习率调度
        # ------------------------------------------------
        if global_step <= warmup_steps:
            # Phase 1: Linear Warmup
            lr = lr_weights_init + (lr_weights_max - lr_weights_init) * (global_step / warmup_steps)
        else:
            # Phase 2 & 3: Step Decay
            lr = lr_weights_max
            for decay_step in lr_decay_steps:
                if global_step >= decay_step:
                    lr *= lr_decay_factor

        # 更新优化器学习率
        for param_group in optimizer_w.param_groups:
            param_group['lr'] = lr

        # 1. 获取数据
        batch_w = next(iter_w)
        batch_a = next(iter_a)

        input_w, label_w = batch_w["image"].to(device), batch_w["label"].to(device)
        input_a, label_a = batch_a["image"].to(device), batch_a["label"].to(device)

        # ------------------------------------------------
        # 阶段 A: 更新架构参数 (Alphas)
        # ------------------------------------------------
        loss_a_val = 0.0
        if global_step > arch_search_start_steps:
            optimizer_a.zero_grad()
            output_a = model(input_a)
            loss_a = loss_function(output_a, label_a)

            probs_children, _ = model.dints_space.get_prob_a(child=True)
            entropy_loss = model.dints_space.get_topology_entropy(probs_children)

            total_loss_a = loss_a + 0.001 * entropy_loss

            total_loss_a.backward()
            optimizer_a.step()
            loss_a_val = total_loss_a.item()

        # ------------------------------------------------
        # 阶段 B: 更新权重参数 (Weights)
        # ------------------------------------------------
        optimizer_w.zero_grad()
        output_w = model(input_w)
        loss_w = loss_function(output_w, label_w)

        loss_w.backward()
        optimizer_w.step()

        status_str = "WARMUP" if global_step <= warmup_steps else \
                     ("STABLE" if global_step <= arch_search_start_steps else "SEARCH")

        if global_step % 10 == 0:
            current_time = time.time()
            step_time = current_time - loop_start_time
            loop_start_time = current_time  # 只在打印后重置
            print(f"Step {global_step}/{max_iterations} [{status_str}] | Time: {step_time:.2f}s | "
                  f"Loss W: {loss_w.item():.4f} | Loss A: {loss_a_val:.4f}")

        # --- 验证与保存 ---
        if global_step % eval_interval == 0:
            torch.cuda.empty_cache()

            val_start_time = time.time()
            model.eval()
            with torch.no_grad():
                val_count = 0
                for val_data in val_loader:
                    if num_val_samples > 0 and val_count >= num_val_samples:
                        break

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

                    val_count += 1

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

            val_duration = time.time() - val_start_time
            print(f"Validation at Step {global_step} | Time: {val_duration:.1f}s | Val Dice: {metric:.4f}", end="")

            if metric > best_metric:
                best_metric = metric
                best_metric_step = global_step
                try:
                    # 获取最佳架构
                    topology = model.get_topology()

                    arch_json = {
                        "arch_code_a": topology[1].tolist(),
                        "arch_code_c": topology[2].tolist(),
                        "arch_code_a_max": topology[3].tolist()
                    }
                    save_path = os.path.join(arch_save_dir, "best_arch_roi.json")
                    with open(save_path, "w") as f:
                        json.dump(arch_json, f, indent=4)

                    print(f" -> 🔥 New Best! Saved arch", end="")
                except Exception as e:
                    print(f" -> [Err] Save Failed: {e}", end="")

            print("")

            torch.cuda.empty_cache()
            model.train()

    print(f"\n训练结束。最佳模型 Dice: {best_metric:.4f} (at Step {best_metric_step})")

if __name__ == "__main__":
    parser = get_config_argument_parser(description="DiNTS Search Script")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
    args = parser.parse_args()

    # 加载配置
    default_config_path = os.path.join(project_root, "configs", "dints_search.yaml")
    config_path = args.config if args.config else default_config_path

    # 不提供默认配置，强制读取 YAML
    config = load_config(config_path, default_config=None)

    if not config:
        print(f"❌ 错误: 无法加载配置文件 {config_path}，或文件为空！")
        sys.exit(1)

    # 命令行参数优先级最高
    if args.seed != 2025:
        config["seed"] = args.seed
    elif "seed" not in config:
        config["seed"] = 2025 # Fallback if not in config and not in args

    try:
        search_baseline(config)
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
