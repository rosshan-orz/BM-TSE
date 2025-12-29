import torch
import torch.nn as nn

class ResConv(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(ResConv, self).__init__()
        # 步骤1: 降维卷积 (in_channels -> 64)
        self.down_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # 步骤2: 非线性激活 (使用 GELU)
        self.gelu = nn.GELU()
        
        # 步骤3: 升维卷积 (64 -> out_channels)
        self.up_conv = nn.Conv1d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        
        # 确保输入和输出通道数相同，以便进行残差连接
        # 如果不相同，需要额外处理，但图片中暗示它们是相同的 (128 -> 128)
        assert in_channels == out_channels, "输入和输出通道数必须相同才能进行残差连接"

    def forward(self, x):
        # 接收输入 x，并保存原始输入用于残差连接
        identity = x
        
        # 执行降维卷积和激活
        out = self.down_conv(x)
        out = self.gelu(out)
        
        # 执行升维卷积
        out = self.up_conv(out)
        
        # 步骤4: 残差连接
        # 将原始输入与模块的输出相加
        out += identity
        
        return out

