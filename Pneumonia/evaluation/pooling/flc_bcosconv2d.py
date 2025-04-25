from .flc_pooling import FLC_Pooling, correct_ASAP_padding_large
#from .flc_pooling import FLC_Pooling_NoHW
from libraries.bcosconv2d import NormedConv2d
from libraries.bcosconv2d import BcosConv2d


import torch.nn as nn


class ModifiedFLCBcosConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), b=2, max_out=1, transpose=False):
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
        self.flcpool = FLC_Pooling(transpose=transpose) if isinstance(stride, tuple) and stride[0] > 1 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flcpool(x)
        return x
    
    
class ModifiedFLCASAPBcosConv2d(nn.Module):
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
        self.flcpool = correct_ASAP_padding_large() if isinstance(stride, tuple) and stride[0] > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.flcpool(x)
        return x



'''class ModifiedFLCBcosConv2dNoHW(nn.Module):
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
        self.flcpool = FLC_Pooling_NoHW() if isinstance(stride, tuple) and stride[0] > 1 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flcpool(x)
        return x'''
