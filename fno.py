"""
Adapted from the original implementation: https://github.com/scaomath/fourier_neural_operator
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

class UpperTriangularParameter(nn.Module):
    def __init__(self, i, o, x, rand_scale=1.0, dtype=torch.float):
        super().__init__()
        self.i, self.o, self.x = i, o, x
        self.num_tri = x * (x + 1) // 2

        # Only store the upper triangular values — no wasted parameters
        self.params = nn.Parameter(torch.randn(self.i, self.o, self.num_tri, dtype=dtype) * rand_scale)

        # Cache indices (not a parameter, just a buffer)
        row_idx, col_idx = torch.triu_indices(self.x, self.x)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def forward(self) -> torch.Tensor:
        """Returns the full (i, o, x, y) tensor with zeros below diagonal."""
        out = torch.zeros(self.i, self.o, self.x, self.x,
                          dtype=self.params.dtype, device=self.params.device)
        out[..., self.row_idx, self.col_idx] = self.params
        return out

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, isotropic=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.isotropic = isotropic

        self.scale = (1 / (in_channels * out_channels))
        
        if isotropic:
            # We only need real values due to x <-> -x symmetry
            self.weights = UpperTriangularParameter(in_channels, out_channels, self.modes1, self.scale, dtype=torch.float)
        else:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        if self.isotropic:
            #Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft2(x).to(torch.cfloat)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

            w = self.weights().to(torch.cfloat)
            # x-y symmetry enforcement
            w = w + w.transpose(2, 3)

            # Top left
            out_ft[:, :, :self.modes1, :self.modes2] += \
                self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w)
            # Bottom left - skip the kx = 0 mode
            out_ft[:, :, -self.modes1+1:, :self.modes2] += \
                self.compl_mul2d(x_ft[:, :, -self.modes1+1:, :self.modes2], w[:, :, 1:, :].flip((2,)).conj())

            #Return to physical space
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))).real
            return x
            
        else:
            # Original model
            #Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft2(x)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            #Return to physical space
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
            return x
            
        


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, isotropic=False):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.isotropic = isotropic
        
        self.fc0 = nn.Linear(1, self.width) # input channel is 1: (a(x, y))

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, isotropic=isotropic)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, isotropic=isotropic)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, isotropic=isotropic)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, isotropic=isotropic)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x, **kwargs):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x