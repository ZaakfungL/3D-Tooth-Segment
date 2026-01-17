import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR


class SwinUNETR3D(nn.Module):
    """
    基于 MONAI 的 SwinUNETR 封装类。
    SwinUNETR 结合了 Swin Transformer 编码器和 CNN 解码器，
    在3D医学图像分割中表现优异，尤其适合捕获长距离依赖关系。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        depths: tuple = (2, 2, 2, 2),
        num_heads: tuple = (3, 6, 12, 24),
        use_checkpoint: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        """
        Args:
            in_channels: 输入通道数 (默认1)。
            out_channels: 输出类别数 (默认2: 背景 + 牙齿)。
            feature_size: 初始特征维度。
            depths: 每个阶段的 Swin Transformer 块数量。
            num_heads: 每个阶段的注意力头数量。
            use_checkpoint: 是否使用梯度检查点以节省显存。
            drop_rate: Dropout 概率。
            attn_drop_rate: Attention Dropout 概率。
        """
        super(SwinUNETR3D, self).__init__()

        self.model = MonaiSwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            spatial_dims=3,
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 单元测试代码 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("正在测试 SwinUNETR3D 模型...")

    model = SwinUNETR3D(in_channels=1, out_channels=2)
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
