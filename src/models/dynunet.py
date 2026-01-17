import torch
import torch.nn as nn
from monai.networks.nets import DynUNet as MonaiDynUNet


class DynUNet3D(nn.Module):
    """
    基于 MONAI 的 DynUNet 封装类。
    DynUNet 是 nnU-Net 的核心架构，通过动态配置卷积核大小和步长，
    自适应不同的输入尺寸和任务需求。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        kernel_size: list = None,
        strides: list = None,
        upsample_kernel_size: list = None,
        filters: list = None,
        deep_supervision: bool = False,
        res_block: bool = True,
    ):
        """
        Args:
            in_channels: 输入通道数 (默认1)。
            out_channels: 输出类别数 (默认2: 背景 + 牙齿)。
            kernel_size: 每层的卷积核大小。
            strides: 每层的步长 (决定下采样倍数)。
            upsample_kernel_size: 上采样卷积核大小。
            filters: 每层的滤波器数量。
            deep_supervision: 是否启用深度监督。
            res_block: 是否使用残差块。
        """
        super(DynUNet3D, self).__init__()

        # 默认配置: 6层编码器
        if kernel_size is None:
            kernel_size = [[3, 3, 3]] * 6
        if strides is None:
            strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        if upsample_kernel_size is None:
            upsample_kernel_size = strides[1:]
        if filters is None:
            filters = [32, 64, 128, 256, 512, 512]

        self.model = MonaiDynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            deep_supervision=deep_supervision,
            res_block=res_block,
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 单元测试代码 (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("正在测试 DynUNet3D 模型...")

    model = DynUNet3D(in_channels=1, out_channels=2)
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
