import sys
import os
import glob
import torch
import time
import warnings
import argparse

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
from src.ssl.tmo import TMOAdamW
from src.utils.config import load_config, get_config_argument_parser

def train_tmo(config):
    # ================= 配置区域 (Fail Fast) =================
    seed = config.get("seed", 2025)
    gpu_id = str(config["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU: {gpu_id}")

    data_dir = config["data_dir"]
    model_save_dir = config["model_save_dir"].format(seed=seed)
    os.makedirs(model_save_dir, exist_ok=True)

    load_batch_size_l = config["load_batch_size_l"]
    num_samples_l = config["num_samples_l"]

    load_batch_size_u = config["load_batch_size_u"]
    num_samples_u = config["num_samples_u"]

    num_labeled_use = config["num_labeled_use"]
    num_unlabeled_use = config["num_unlabeled_use"]

    max_iterations = config["max_iterations"]
    val_interval = config["val_interval"]

    lr = config["lr"]
    roi_size = tuple(config["roi_size"])

    ema_decay = config["ema_decay"]
    consistency = config["consistency"]
    # 动态计算或从配置读取
    consistency_rampup = config.get("consistency_rampup", max_iterations // 5)

    num_workers = config["num_workers"]
    cache_rate = config["cache_rate"]

    # ================= 1. 数据准备 =================
    set_determinism(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始 TMO (Trusted Momentum) 训练 | 设备: {device}")
    print(f"📌 总计 {max_iterations} Iterations")
    print(f"当前随机种子: {seed}")

    # A. 准备有标签数据 (Labeled)
    labeled_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labeled_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    labeled_dicts = [{"image": i, "label": l} for i, l in zip(labeled_images, labeled_labels)]

    # 划分 Train/Val
    train_labeled_files, val_files = partition_dataset(
        data=labeled_dicts, ratios=[0.8, 0.2], shuffle=True, seed=seed
    )

    # B. 准备无标签数据 (Unlabeled)
    unlabeled_dir = os.path.join(data_dir, "imagesUnlabeled")
    if os.path.exists(unlabeled_dir):
        unlabeled_images = sorted(glob.glob(os.path.join(unlabeled_dir, "*.nii.gz")))
        unlabeled_dicts = [{"image": i, "label": i} for i in unlabeled_images]

        import random
        # 确保随机性受控
        random.seed(seed)
        random.shuffle(unlabeled_dicts)
        unlabeled_dicts = unlabeled_dicts[:num_unlabeled_use]
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
        batch_size=load_batch_size_l,
        roi_size=roi_size,
        num_samples=num_samples_l,
        is_train=True,
        num_workers=num_workers,
        cache_rate=cache_rate,
    )

    # 2. 无标签加载器
    loader_unlabeled = get_basic_loader(
        data_list=unlabeled_dicts,
        batch_size=load_batch_size_u,
        roi_size=roi_size,
        num_samples=num_samples_u,
        is_train=True,
        num_workers=num_workers,
        cache_rate=cache_rate,
    )

    # 3. 验证加载器
    loader_val = get_basic_loader(
        data_list=val_files,
        batch_size=1,
        roi_size=roi_size,
        is_train=False,
        num_workers=num_workers,
        cache_rate=cache_rate,
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
    optimizer = TMOAdamW(model.parameters(), lr=lr)

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

    while iteration < max_iterations:
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

        consistency_weight = get_current_consistency_weight(iteration, max_iterations, consistency, consistency_rampup)

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
        update_ema_variables(model, ema_model, ema_decay, iteration)

        # --- Logging ---
        current_time = time.time()
        iter_time = current_time - loop_start_time
        loop_start_time = current_time

        if iteration % 10 == 0:
            print(f"Iter {iteration}/{max_iterations} | Time: {iter_time:.4f}s | "
                  f"L_Sup: {l_sup.item():.4f} | L_Cons: {l_cons.item():.4f} (w={consistency_weight:.3f})")

        # --- Validation (使用 Teacher 评估) ---
        if iteration % val_interval == 0:
            torch.cuda.empty_cache()

            ema_model.eval()
            with torch.no_grad():
                for val_data in loader_val:
                    val_in, val_lbl = val_data["image"].to(device), val_data["label"].to(device)
                    val_out = sliding_window_inference(val_in, roi_size, 4, ema_model)
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
    parser = get_config_argument_parser(description="UNet3D TMO 半监督训练脚本")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子 (默认: 2025)")
    args = parser.parse_args()

    # 加载配置
    default_config_path = os.path.join(project_root, "configs", "unet3D_tmo_train.yaml")
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
        train_tmo(config)
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
