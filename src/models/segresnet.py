import torch
import torch.nn as nn
from monai.networks.nets import SegResNet as MonaiSegResNet


class SegResNet3D(nn.Module):
    """
    基于 MONAI 的 SegResNet 封装类。
    SegResNet 是一个基于残差连接的编码器-解码器架构，
    专为3D医学图像分割设计，参数量适中且性能优秀。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        init_filters: int = 32,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        dropout_prob: float = 0.0,
    ):
        """
        Args:
            in_channels: 输入通道数 (默认1，单模态CT/MRI)。
            out_channels: 输出类别数 (默认2: 背景 + 牙齿)。
            init_filters: 初始卷积滤波器数量。
            blocks_down: 编码器每个阶段的残差块数量。
            blocks_up: 解码器每个阶段的残差块数量。
            dropout_prob: Dropout 概率。
        """
        super(SegResNet3D, self).__init__()

        self.model = MonaiSegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 单元测试代码 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("正在测试 SegResNet3D 模型...")

    model = SegResNet3D(in_channels=1, out_channels=2)
    input_tensor = torch.randn(2, 1, 96, 96, 96)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")

    if output.shape == (2, 2, 96, 96, 96):
        print("测试通过! 模型结构正确。")
    else:
        print("测试失败! 输出尺寸不匹配。")
