import os
import sys
import glob
import itertools
from monai.data import partition_dataset

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    
from src.dataloaders.basic_loader import get_basic_loader

class NASComboDataLoader:
    """
    DINTS-TMO 专用组合数据加载器 (The Quad-Stream Loader)
    
    功能：
    将数据集划分为 4 个互斥的子集，并提供同步的迭代器：
    1. Labeled_Weights (L_W): 用于更新卷积权重 (TMO Labeled Step)
    2. Unlabeled_Weights (U_W): 用于更新卷积权重 (TMO Unlabeled Step)
    3. Labeled_Arch (L_A): 用于更新架构参数 (TMO Labeled Step)
    4. Unlabeled_Arch (U_A): 用于更新架构参数 (TMO Unlabeled Step)
    
    迭代策略：
    - 无标签数据集 (U_W, U_A) 较大，决定 Epoch 的长度。
    - 有标签数据集 (L_W, L_A) 较小，会无限循环 (cycle) 以匹配无标签数据的步数。
    """
    def __init__(
        self,
        data_dir,
        batch_size_l=2,
        batch_size_u=2,
        roi_size=(96, 96, 96),
        num_workers=2,
        cache_rate=0.0,
        seed=2025,
        limit=None # Debug用
    ):
        self.data_dir = data_dir
        self.batch_size_l = batch_size_l
        self.batch_size_u = batch_size_u
        
        # ================= 1. 原始数据扫描 =================
        print(f"📦 初始化 NAS-TMO 组合加载器...")
        
        # A. 扫描有标签数据
        images_l = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        labels_l = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        dicts_l = [{"image": i, "label": l} for i, l in zip(images_l, labels_l)]
        
        # B. 扫描无标签数据
        images_u = sorted(glob.glob(os.path.join(data_dir, "imagesUnlabeled", "*.nii.gz")))
        if len(images_u) == 0:
            print("⚠️ 警告: 未找到 imagesUnlabeled，将在无标签流中复用有标签数据 (伪SSL模式)")
            dicts_u = [{"image": i, "label": i} for i in images_l] # 复用
        else:
            dicts_u = [{"image": i, "label": i} for i in images_u]

        # [Debug] 限制数据量
        if limit is not None:
            dicts_l = dicts_l[:limit]
            dicts_u = dicts_u[:limit]
            print(f"⚡ [Debug] 数据量限制为: {limit}")

        # ================= 2. 四分法切割 (Quad-Split) =================
        # 这里的 shuffle=True 配合 seed 保证了每次实验的切分是一致的
        
        # 切分有标签数据 (50% Weights, 50% Arch)
        # 注意：这里我们先假设外部已经分好了 Train/Val，传入的 data_dir 应该只包含 Train 部分
        # 如果不是，建议在外部先做一次 Train/Val Split，只把 Train 列表传进来。
        # 但为了通用性，这里假设 dicts_l 就是全部可用的训练数据。
        l_w, l_a = partition_dataset(
            data=dicts_l, ratios=[0.5, 0.5], shuffle=True, seed=seed
        )
        
        # 切分无标签数据 (50% Weights, 50% Arch)
        u_w, u_a = partition_dataset(
            data=dicts_u, ratios=[0.5, 0.5], shuffle=True, seed=seed
        )
        
        print(f"📊 数据划分完成 (Quad-Split):")
        print(f"   [Phase Weights] Labeled: {len(l_w)} | Unlabeled: {len(u_w)}")
        print(f"   [Phase Arch   ] Labeled: {len(l_a)} | Unlabeled: {len(u_a)}")

        # ================= 3. 创建 4 个基础加载器 =================
        # 这里的 num_samples=1 是为了 NAS 显存优化
        common_args = dict(
            roi_size=roi_size, 
            is_train=True, 
            num_workers=num_workers, 
            cache_rate=cache_rate,
            num_samples=1 
        )

        self.loader_l_w = get_basic_loader(data_list=l_w, batch_size=batch_size_l, **common_args)
        self.loader_u_w = get_basic_loader(data_list=u_w, batch_size=batch_size_u, **common_args)
        
        self.loader_l_a = get_basic_loader(data_list=l_a, batch_size=batch_size_l, **common_args)
        self.loader_u_a = get_basic_loader(data_list=u_a, batch_size=batch_size_u, **common_args)
        
        # 计算一个 Epoch 的步数 (以较大的无标签数据集为准)
        self.steps_per_epoch = min(len(self.loader_u_w), len(self.loader_u_a))

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        """
        生成器：每次 yield 一个包含 4 部分数据的字典。
        有标签数据会无限循环 (cycle)，直到无标签数据遍历完毕。
        """
        # 组合迭代器
        # cycle() 让有标签数据用完后重头开始
        iterator = zip(
            itertools.cycle(self.loader_l_w),
            self.loader_u_w,
            itertools.cycle(self.loader_l_a),
            self.loader_u_a
        )
        
        for batch_l_w, batch_u_w, batch_l_a, batch_u_a in iterator:
            # 打包返回
            yield {
                "l_w": batch_l_w, # Labeled for Weights
                "u_w": batch_u_w, # Unlabeled for Weights
                "l_a": batch_l_a, # Labeled for Arch
                "u_a": batch_u_a  # Unlabeled for Arch
            }


# --- 单元测试代码 ---
if __name__ == "__main__":
    
    # 假设你的数据在这里
    TEST_DATA_DIR = "/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI"
    
    if os.path.exists(TEST_DATA_DIR):
        print("=" * 60)
        print("正在运行 NASComboDataLoader 单元测试...")
        print("=" * 60)
        
        print("\n>>> 测试 Quad-Stream Loader (limit=4):")
        combo_loader = NASComboDataLoader(
            data_dir=TEST_DATA_DIR,
            batch_size_l=2,
            batch_size_u=2,
            roi_size=(64, 64, 64),
            num_workers=0,
            cache_rate=0.0,
            seed=2025,
            limit=8
        )
        
        print(f"\n📏 每个 Epoch 的步数: {len(combo_loader)}")
        print("\n🚀 开始迭代测试 (最多显示 3 个 batch)...")
        
        count = 0
        for batch in combo_loader:
            count += 1
            print(f"\n--- Batch {count} ---")
            print(f"  [L_W] Labeled for Weights   - Image: {batch['l_w']['image'].shape}, Label: {batch['l_w']['label'].shape}")
            print(f"  [U_W] Unlabeled for Weights - Image: {batch['u_w']['image'].shape}")
            print(f"  [L_A] Labeled for Arch      - Image: {batch['l_a']['image'].shape}, Label: {batch['l_a']['label'].shape}")
            print(f"  [U_A] Unlabeled for Arch    - Image: {batch['u_a']['image'].shape}")
            
            if count >= 3:
                print("\n⏸️ 已显示 3 个 batch，提前退出...")
                break
        
        print(f"\n✅ NASComboDataLoader 测试完成！共迭代 {count} 个 batch")
        print("=" * 60)
    else:
        print(f"提示: {TEST_DATA_DIR} 不存在，跳过测试。")
        print("请修改 TEST_DATA_DIR 为你的数据集路径。")