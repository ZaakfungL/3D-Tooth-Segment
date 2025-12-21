import torch
import torch.nn as nn
from monai.utils import optional_import
from monai.networks.nets import DiNTS as MonaiDiNTS
from monai.networks.nets import TopologySearch

class DiNTSWrapper(nn.Module):
    """
    DiNTS 搜索空间的封装器。
    用于 NAS 搜索阶段 (Stage 1 & Stage 2 Search)。
    """
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 num_blocks: int = 6,       # [显存优化] 默认减小为 6
                 num_depths: int = 3,       # [显存优化] 默认减小为 3
                 use_downsample: bool = True,
                 spatial_dims: int = 3,
                 p_dropout: float = 0.1,
                 channel_mul: int = 1):
        super(DiNTSWrapper, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. 实例化拓扑搜索空间 (TopologySearch)
        self.dints_space = TopologySearch(
            spatial_dims=spatial_dims,
            num_blocks=num_blocks,
            num_depths=num_depths,
            use_downsample=use_downsample,
            channel_mul=channel_mul,
        )

        # 2. 定义 DiNTS 超网络
        self.dints_search = MonaiDiNTS(
            dints_space=self.dints_space,
            in_channels=in_channels,
            num_classes=out_channels,
            act_name="RELU",
            norm_name="INSTANCE",
            use_downsample=use_downsample,
            spatial_dims=spatial_dims,
        )

    def forward(self, x):
        """
        前向传播。
        注意：DiNTS 内部已经管理了架构参数 (log_alpha_a, log_alpha_c) 的使用。
        """
        return self.dints_search(x)

    def arch_parameters(self):
        """
        返回架构参数 (Alphas)，用于架构优化器 (Arch Optimizer)。
        TopologySearch 内部有两个架构参数:
        - log_alpha_a: 宏观路径权重
        - log_alpha_c: 微观操作权重
        """
        return [self.dints_space.log_alpha_a, self.dints_space.log_alpha_c]

    def weight_parameters(self):
        """
        返回权重参数 (Weights)，用于权重优化器 (Weight Optimizer)。
        排除掉架构参数。
        """
        # 收集所有参数
        all_params = list(self.parameters())
        # 收集架构参数的 ID
        arch_ids = list(map(id, self.arch_parameters()))
        # 过滤
        return [p for p in all_params if id(p) not in arch_ids]

    def get_topology(self):
        """
        解码最终架构。
        """
        # TopologySearch 提供了 decode 方法，返回 (node_a, arch_code_a, arch_code_c, arch_code_a_max)
        return self.dints_space.decode()

# ==========================================
# 单元测试代码
# ==========================================
if __name__ == "__main__":
    print("🚀 正在测试 DiNTSWrapper (Search Mode)...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: CUDA")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")
    
    try:
        # 1. 初始化模型
        model = DiNTSWrapper(
            in_channels=1, 
            out_channels=2, 
            num_blocks=6, 
            num_depths=3
        ).to(device)
        print("✅ 模型初始化成功")
        
        # 2. 检查架构参数
        arch_params = model.arch_parameters()
        print(f"架构参数数量: {len(arch_params)} (应为 2: log_alpha_a, log_alpha_c)")
        print(f"log_alpha_a shape: {arch_params[0].shape}")
        print(f"log_alpha_c shape: {arch_params[1].shape}")
        
        # 3. 前向传播测试
        # DiNTS 的输入要求比较特殊，通常需要符合 2^num_depths 的倍数
        input_tensor = torch.randn(2, 1, 64, 64, 64).to(device)
        output = model(input_tensor)
        print(f"输入 Shape: {input_tensor.shape}")
        print(f"输出 Shape: {output.shape}") 
        
        if output.shape == (2, 2, 64, 64, 64):
            print("✅ 前向传播形状匹配")
        else:
            print(f"❌ 前向传播形状错误: {output.shape}")

        # 4. 反向传播测试 (验证 Alpha 梯度)
        loss = output.sum()
        loss.backward()
        
        # 检查架构参数是否有梯度
        if model.dints_space.log_alpha_a.grad is not None:
            print("✅ 架构参数 (Alpha) 梯度计算成功")
        else:
            print("❌ 架构参数 (Alpha) 无梯度！")
            
        print("🎉 所有测试通过！")

    except Exception as e:
        print(f"\n❌ 测试发生异常: {e}")
        import traceback
        traceback.print_exc()