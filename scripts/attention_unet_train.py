import sys
import os
import glob
import torch
import time
import warnings

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

from src.models.attention_unet import AttentionUNet3D
from src.dataloaders.basic_loader import get_basic_loader
from src.utils.config import load_config, get_config_argument_parser

def train_attention_unet(config):
    # ================= 配置区域 =================
    seed = config.get("seed", 2025)
    gpu_id = str(config["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU: {gpu_id}")

    data_dir = config["data_dir"]
    model_save_dir = config["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)

    max_iterations = config["max_iterations"]
    val_interval = config["val_interval"]

    load_batch_size = config["load_batch_size"]
    num_samples = config["num_samples"]

    lr = config["lr"]
    roi_size = tuple(config["roi_size"])

    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    # ================= 1. 数据准备 =================
    set_determinism(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"当前随机种子: {seed}")

    print("正在扫描并划分数据集...")
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    if not images:
        raise ValueError(f"错误：在 {data_dir} 中未找到数据！")

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

    train_files, val_files = partition_dataset(
        data=data_dicts,
        ratios=[0.8, 0.2],
        shuffle=True,
        seed=seed
    )

    print(f"  - 总数据量: {len(data_dicts)}")
    print(f"  - 训练集 (80%): {len(train_files)} 例")
    print(f"  - 验证集 (20%): {len(val_files)} 例")

    # ================= 2. 创建加载器 =================
    train_loader = get_basic_loader(
        data_list=train_files,
        batch_size=load_batch_size,
        roi_size=roi_size,
        num_samples=num_samples,
        is_train=True,
        num_workers=num_workers,
        cache_rate=cache_rate,
    )

    val_loader = get_basic_loader(
        data_list=val_files,
        batch_size=1,
        roi_size=roi_size,
        is_train=False,
        num_workers=num_workers,
        cache_rate=cache_rate,
    )

    # ================= 3. 模型与优化器 =================
    model = AttentionUNet3D(in_channels=1, out_channels=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ================= 4. 训练循环 =================
    best_metric = -1
    best_metric_iteration = -1
    iteration = 0
    epoch_loss = 0
    step_in_epoch = 0

    print(f"\n{'='*20} 开始训练 AttentionUNet (基于Iteration) {'='*20}")
    print(f"最大迭代次数: {max_iterations}, 验证间隔: {val_interval} iterations")

    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()
    loop_start_time = time.time()

    while iteration < max_iterations:
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

        current_loss = loss.item()

        if iteration % 10 == 0:
            current_time = time.time()
            iter_time = current_time - loop_start_time
            loop_start_time = current_time
            print(f"Iteration {iteration}/{max_iterations} | Time: {iter_time:.2f}s | Loss: {current_loss:.4f}")

        # --- Validation ---
        if iteration % val_interval == 0:
            # 释放训练阶段的中间变量
            del inputs, labels_batch, outputs, loss
            torch.cuda.empty_cache()
            val_start_time = time.time()

            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=roi_size,
                        sw_batch_size=16,
                        predictor=model
                    )

                    val_outputs = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_outputs)]
                    val_labels = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs, y=val_labels)

                    del val_inputs, val_labels, val_outputs

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                val_time = time.time() - val_start_time

                print(f"Validation at Iter {iteration} | Val Dice: {metric:.4f} | Val Time: {val_time:.2f}s", end="")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_iteration = iteration
                    save_path = os.path.join(model_save_dir, f"best_model_{seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f" -> New Best! ({best_metric:.4f})")
                else:
                    print("")

            torch.cuda.empty_cache()

            model.train()
            epoch_loss = 0
            step_in_epoch = 0

    total_time = time.time() - start_time
    print(f"\n训练结束。总用时: {total_time:.1f}s")
    print(f"最佳模型 Dice: {best_metric:.4f} 于 Iteration {best_metric_iteration}")

if __name__ == "__main__":
    parser = get_config_argument_parser(description="AttentionUNet 训练脚本")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子 (默认: 2025)")
    args = parser.parse_args()

    default_config_path = os.path.join(project_root, "configs", "attention_unet_train.yaml")
    config_path = args.config if args.config else default_config_path

    config = load_config(config_path, default_config=None)

    if not config:
        print(f"错误: 无法加载配置文件 {config_path}，或文件为空！")
        sys.exit(1)

    if args.seed != 2025:
        config["seed"] = args.seed
    elif "seed" not in config:
        config["seed"] = 2025

    try:
        train_attention_unet(config)
    except Exception as e:
        print(f"训练发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
