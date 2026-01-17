import torch
import torch.nn as nn
from monai.networks.nets import UNETR as MonaiUNETR


class UNETR3D(nn.Module):
    """
    基于 MONAI 的 UNETR 封装类。
    UNETR 使用 Vision Transformer (ViT) 作为编码器，
    结合 CNN 解码器进行3D医学图像分割。
    适合捕获全局上下文信息。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        img_size: tuple = (96, 96, 96),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            in_channels: 输入通道数 (默认1)。
            out_channels: 输出类别数 (默认2: 背景 + 牙齿)。
            img_size: 输入图像大小，必须与 roi_size 一致。
            feature_size: CNN 解码器的初始特征维度。
            hidden_size: Transformer 隐藏层维度。
            mlp_dim: Transformer MLP 维度。
            num_heads: 多头注意力的头数。
            dropout_rate: Dropout 概率。
        """
        super(UNETR3D, self).__init__()

        self.model = MonaiUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 单元测试代码 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("正在测试 UNETR3D 模型...")

    model = UNETR3D(in_channels=1, out_channels=2, img_size=(96, 96, 96))
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
