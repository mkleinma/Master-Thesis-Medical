'''FLC Pooling module
can be used and distributed under the MIT license
Reference:
[1] Grabinski, J., Jung, S., Keuper, J., & Keuper, M. (2022). 
    "FrequencyLowCut Pooling--Plug & Play against Catastrophic Overfitting." 
    European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
'''

import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

class FLC_Pooling(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, transpose=True):
        self.transpose = transpose
        self.window2d = None
        super(FLC_Pooling, self).__init__()

    def forward(self, x):        
        #x = x.cuda()
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        x = x.to(torch.float32).to(device)
        #import ipdb;ipdb.set_trace()
        if self.window2d is None:
            size=x.size(2)
            window1d = np.abs(np.hamming(size))
            window2d = np.sqrt(np.outer(window1d,window1d))
            window2d = torch.Tensor(window2d).to(device)            
            self.window2d = window2d.unsqueeze(0).unsqueeze(0)

        orig_x_size = x.shape
        x = F.pad(x, (x.shape[-1]//2, x.shape[-1]//2, x.shape[-2]//2, x.shape[-2]//2)).to(device) # geändert zu 2 wegen kosten


        if self.transpose:
            x = x.transpose(2,3)
        
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        #low_part = low_part.cuda()*self.window2d
        try:
            assert low_part.size(2) == self.window2d.size(2)
            assert low_part.size(3) == self.window2d.size(3)
            low_part = low_part*self.window2d
        except Exception:
            try:
                assert low_part.size(2) == self.window2d.size(2)
                assert low_part.size(3) == self.window2d.size(3)
                low_part = low_part.cuda()*self.window2d.cuda()
            except Exception:
                #import ipdb;ipdb.set_trace()
                window1d = np.abs(np.hamming(x.shape[2]))
                window1d_2 = np.abs(np.hamming(x.shape[3]))
                window2d = np.sqrt(np.outer(window1d,window1d_2))
                window2d = torch.Tensor(window2d).to(device)
                self.window2d = window2d.unsqueeze(0).unsqueeze(0)
                low_part = low_part.to(device)*self.window2d.to(device)
        
        #low_part = low_part[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        low_part = low_part[:,:,int(orig_x_size[2]/4):int(orig_x_size[2]/4*3),int(orig_x_size[3]/4):int(orig_x_size[3]/4*3)]
        
        return torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real.half().to(device)
