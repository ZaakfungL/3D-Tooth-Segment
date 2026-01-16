import torch
import torch.nn as nn
from monai.networks.nets import VNet as MonaiVNet

class VNet3D(nn.Module):
    """
    基于 MONAI 的 3D V-Net 封装类。
    V-Net 使用残差连接和体积卷积，专为3D医学图像分割设计。
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 dropout_prob_up: tuple = (0.0, 0.0),
                 dropout_prob_down: float = 0.0,
                 bias: bool = False):
        """
        Args:
            in_channels: 输入通道数。注意: VNet 要求 16 % in_channels == 0。
            out_channels: 输出类别数 (例如 2: 背景 + 牙齿)。
            dropout_prob_up: 上采样路径的 Dropout 概率 (2 元组)。
            dropout_prob_down: 下采样路径的 Dropout 概率 (单个浮点数)。
            bias: 是否使用偏置项。
        """
        super(VNet3D, self).__init__()

        # 验证 VNet 的输入通道约束
        if 16 % in_channels != 0:
            raise ValueError(
                f"VNet 要求 16 % in_channels == 0，但收到 in_channels={in_channels}"
            )

        self.model = MonaiVNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob_up=dropout_prob_up,
            dropout_prob_down=dropout_prob_down,
            bias=bias
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 单元测试代码 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("正在测试 VNet3D 模型...")

    # 测试配置
    model = VNet3D(in_channels=1, out_channels=2)
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
