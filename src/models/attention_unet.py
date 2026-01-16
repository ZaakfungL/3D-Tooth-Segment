import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet as MonaiAttentionUnet

class AttentionUNet3D(nn.Module):
    """
    基于 MONAI 的 3D Attention U-Net 封装类。
    Attention U-Net 通过注意力门控机制增强特征选择能力。
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 channels: tuple = (32, 64, 128, 256, 512),
                 strides: tuple = (2, 2, 2, 2),
                 kernel_size: int = 3,
                 up_kernel_size: int = 3,
                 dropout: float = 0.0):
        """
        Args:
            in_channels: 输入通道数。
            out_channels: 输出类别数 (例如 2: 背景 + 牙齿)。
            channels: 每一层的通道数，长度必须比 strides 多 1。
            strides: 下采样的步长。
            kernel_size: 卷积核大小。
            up_kernel_size: 上采样卷积核大小。
            dropout: Dropout 比率。
        """
        super(AttentionUNet3D, self).__init__()

        # 验证 channels 和 strides 长度关系
        if len(channels) != len(strides) + 1:
            raise ValueError(
                f"channels 长度 ({len(channels)}) 必须比 strides 长度 ({len(strides)}) 多 1"
            )

        self.model = MonaiAttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            dropout=dropout
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 单元测试代码 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("正在测试 AttentionUNet3D 模型...")

    # 测试配置
    model = AttentionUNet3D(in_channels=1, out_channels=2)
    input_tensor = torch.randn(2, 1, 96, 96, 96)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # 前向传播测试
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    if output.shape == (2, 2, 96, 96, 96):
        print("测试通过！模型结构正确。")
    else:
        print("测试失败！输出尺寸不匹配。")
