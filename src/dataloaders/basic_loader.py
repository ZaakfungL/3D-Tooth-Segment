import os
import glob
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    RandShiftIntensityd,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, Dataset

def get_basic_loader(
    data_dir=None,     # [修改] 改为可选参数
    data_list=None,    # [新增] 支持直接传入划分好的文件列表
    batch_size=2, 
    roi_size=(96, 96, 96), 
    is_train=True, 
    num_workers=2,     # [修改] 默认为 2，防止 WSL 崩溃
    cache_rate=0.0     # [修改] 默认为 0.0 (不缓存)，防止 OOM
):
    """
    基础数据加载器 (Baseline Loader)。
    支持通过 'data_list' 传入划分好的数据集，避免训练集和验证集重叠。

    Args:
        data_dir (str, optional): 数据集根目录。如果不传 data_list，则必须传此参数（自动扫描全量）。
        data_list (list, optional): 包含 {'image': path, 'label': path} 的字典列表。
                                    如果传入此参数，将忽略 data_dir 的扫描。
        batch_size (int): 批次大小。
        roi_size (tuple): 训练时的 Patch 大小，默认 (96, 96, 96)。
        is_train (bool): 是否为训练模式。
        num_workers (int): 多进程加载数。建议在 WSL 中设为 0 或 2。
        cache_rate (float): 缓存比率 (0.0 - 1.0)。内存不足时建议设为 0.0。
    """

    # 1. 确定数据源
    if data_list is not None:
        # 【优先】使用传入的列表 (用于 Train/Val 划分)
        data_dicts = data_list
        source_info = "传入的文件列表 (List)"
    elif data_dir is not None:
        # 【后备】自动扫描目录 (用于简单测试或全量训练)
        # 假设结构是 nnU-Net 原生格式：imagesTr 存放图像，labelsTr 存放标签
        images_dir = os.path.join(data_dir, "imagesTr")
        labels_dir = os.path.join(data_dir, "labelsTr")
        
        # 查找所有 .nii.gz 文件
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
        label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

        # 简单的完整性检查
        if len(image_paths) == 0:
            raise ValueError(f"在 {images_dir} 中未找到任何数据！")
        if len(image_paths) != len(label_paths):
            raise ValueError("图像和标签数量不匹配！请检查文件夹。")

        data_dicts = [
            {"image": img, "label": lbl} 
            for img, lbl in zip(image_paths, label_paths)
        ]
        source_info = f"目录自动扫描: {data_dir}"
    else:
        raise ValueError("错误: 必须提供 data_dir 或 data_list 其中之一！")
    
    print(f"\n[Dataloader] 数据加载配置:")
    print(f"  - 来源: {source_info}")
    print(f"  - 模式: {'训练 (Training)' if is_train else '验证 (Validation)'}")
    print(f"  - 数据量: {len(data_dicts)} 例")
    print(f"  - Patch 尺寸: {roi_size}")
    print(f"  - Workers: {num_workers}, Cache: {cache_rate}")

    # 2. 定义 Transforms (数据增强与预处理)
    if is_train:
        transforms = Compose([
            # --- 基础加载 ---
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"), # 统一方向
            
            # --- 预处理 ---
            # 1. 归一化
            ScaleIntensityd(keys=["image"]), 
            
            # 2. 裁剪前景
            CropForegroundd(keys=["image", "label"], source_key="image"),
            
            # 3. 随机切块 (Patching)
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2, # 每个 volumetric 数据采 2 个 patch
                image_key="image",
                image_threshold=0,
            ),
            
            # --- 数据增强 ---
            RandRotate90d(keys=["image", "label"], prob=0.1, spatial_axes=[0, 2]),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
            
            EnsureTyped(keys=["image", "label"]),
        ])
    else:
        # 验证集 Transforms (不做随机裁剪)
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ])

    # 3. 构建 Dataset 和 DataLoader
    if cache_rate > 0:
        ds = CacheDataset(
            data=data_dicts, 
            transform=transforms, 
            cache_rate=cache_rate, 
            num_workers=num_workers
        )
    else:
        # 使用普通 Dataset (每次从磁盘读取，省内存)
        ds = Dataset(data=data_dicts, transform=transforms)

    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers,
        pin_memory=True # 加速 GPU 传输
    )

    return loader

# --- 单元测试代码 ---
if __name__ == "__main__":
    # 假设你的数据在这里
    TEST_DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI" 
    
    if os.path.exists(TEST_DATA_DIR):
        print("正在运行 DataLoader 单元测试...")
        # 测试: 强制单进程和无缓存，确保不会崩溃
        train_loader = get_basic_loader(data_dir=TEST_DATA_DIR, batch_size=2, is_train=True, num_workers=0, cache_rate=0.0)
        
        check_data = next(iter(train_loader))
        img = check_data["image"]
        print(f"✅ 加载成功! 图像形状: {img.shape}")
    else:
        print(f"提示: {TEST_DATA_DIR} 不存在，跳过测试。")