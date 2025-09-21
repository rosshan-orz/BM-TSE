import torch
import torch.nn as nn

class ResConv(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(ResConv, self).__init__()
        self.down_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.gelu = nn.GELU()
        
        self.up_conv = nn.Conv1d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        assert in_channels == out_channels, "The input and output channels must be the same for residual connection"

    def forward(self, x):
        identity = x
        
        out = self.down_conv(x)
        out = self.gelu(out)
        out = self.up_conv(out)
        out += identity
        
        return out

