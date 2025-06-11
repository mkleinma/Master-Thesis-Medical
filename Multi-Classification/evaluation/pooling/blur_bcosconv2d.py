from antialiased_cnns.blurpool import BlurPool
from libraries_multilabel.bcosconv2d import NormedConv2d
from libraries_multilabel.bcosconv2d import BcosConv2d


import torch.nn as nn


class ModifiedBcosConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), b=2, max_out=1):
        super().__init__()
        # Use BcosConv2d instead of NormedConv2d
        self.conv = BcosConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,  # Set stride=1 for the convolution
            padding=padding,
            b=b,  # Pass the parameter `b` to BcosConv2d
            max_out=max_out,  # Pass the parameter `max_out` to BcosConv2d
        )
        # Apply BlurPool only if stride > 1
        self.blurpool = BlurPool(out_channels, stride=stride[0]) if isinstance(stride, tuple) and stride[0] > 1 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.blurpool(x)
        return x
