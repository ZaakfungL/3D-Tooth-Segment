import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet

class UNet3D(nn.Module):
    """
    基于 MONAI 的 3D U-Net 封装类。
    可以灵活配置用于 Stage 1 (粗分割) 或 Stage 2 (精分割)。
    """
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 features: tuple = (32, 64, 128, 256, 512, 1024),
                 strides: tuple = (2, 2, 2, 2, 2),
                 num_res_units: int = 2,
                 norm: str = "INSTANCE",
                 dropout: float = 0.0):
        """
        Args:
            in_channels: 输入通道数。Stage 1 为 1 (Image), Stage 2 为 2 (Image + Mask).
            out_channels: 输出类别数 (例如 2: 背景 + 牙齿).
            features: 每一层的通道数。
            strides: 下采样的步长。
            num_res_units: 每个层级内部的残差单元数量 (ResU-Net)。
            norm: 归一化方式，推荐 'INSTANCE' 用于医学图像 (Batch Size较小时更稳定)。
            dropout: Dropout 比率。
        """
        super(UNet3D, self).__init__()

        self.model = MonaiUNet(
            spatial_dims=3,          # 这是一个 3D 网络
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout
        )

    def forward(self, x):
        # MONAI UNet 直接返回 logits (未经过 softmax/sigmoid)
        return self.model(x)

# ==========================================
# 单元测试代码 (Unit Test)
# 运行此文件以检查模型是否能正常工作
# ==========================================
if __name__ == "__main__":
    print("正在测试 UNet3D 模型...")
    
    # 1. 模拟 Stage 1 (粗分割) 配置
    # 输入: [Batch, 1, 96, 96, 96], 输出: [Batch, 2, 96, 96, 96]
    stage1_model = UNet3D(in_channels=1, out_channels=2)
    input_tensor_s1 = torch.randn(2, 1, 96, 96, 96) # (B, C, D, H, W)
    
    # 2. 模拟 Stage 2 (精分割) 配置
    # 输入: [Batch, 2, 96, 96, 96] (图像 + 粗掩码), 输出: [Batch, 2, 96, 96, 96]
    stage2_model = UNet3D(in_channels=2, out_channels=2)
    input_tensor_s2 = torch.randn(2, 2, 96, 96, 96)

    # 3. 移动到 GPU (如果有)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_model.to(device)
    stage2_model.to(device)
    input_tensor_s1 = input_tensor_s1.to(device)
    input_tensor_s2 = input_tensor_s2.to(device)

    # 4. 前向传播测试
    with torch.no_grad():
        output_s1 = stage1_model(input_tensor_s1)
        output_s2 = stage2_model(input_tensor_s2)

    print(f"Stage 1 Output Shape: {output_s1.shape}") # 期望: torch.Size([2, 2, 96, 96, 96])
    print(f"Stage 2 Output Shape: {output_s2.shape}") # 期望: torch.Size([2, 2, 96, 96, 96])
    
    if output_s1.shape == (2, 2, 96, 96, 96) and output_s2.shape == (2, 2, 96, 96, 96):
        print("✅ 测试通过！模型结构正确。")
    else:
        print("❌ 测试失败！输出尺寸不匹配。")