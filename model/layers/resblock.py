from typing import Callable
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.layers.conv1d import Conv1DNet, init_weights

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5),):
        super(ResBlock1, self).__init__()
        self.convs1: list[Conv1DNet] = nn.ModuleList([
            Conv1DNet(channels, channels, kernel_size, stride=1, dilation=dilation[0]).weight_norm(),
            Conv1DNet(channels, channels, kernel_size, stride=1, dilation=dilation[1]).weight_norm(),
            Conv1DNet(channels, channels, kernel_size, stride=1, dilation=dilation[2]).weight_norm(),
        ])
        self.convs2: list[Conv1DNet] = nn.ModuleList([
            Conv1DNet(channels, channels, kernel_size, stride=1, dilation=dilation[0]).weight_norm(),
            Conv1DNet(channels, channels, kernel_size, stride=1, dilation=dilation[1]).weight_norm(),
            Conv1DNet(channels, channels, kernel_size, stride=1, dilation=dilation[2]).weight_norm(),
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.silu(x))
            xt = c2(F.silu(xt))
            x = xt + x
        return x
    
    def remove_parametrizations(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            c1.remove_weight_norm()
            c2.remove_weight_norm()

def get_padding(kernel_size: int, dilation: int=1) -> int: return int((kernel_size*dilation-dilation)/2)

class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1,3), activation: Callable[[Tensor], Tensor]=partial(F.silu, inplace=True)):
        super(ResBlock2, self).__init__()
        self.convs: list[Conv1DNet] = nn.ModuleList([
            Conv1DNet(channels, channels, kernel_size, stride=1, padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0]).weight_norm(),
            Conv1DNet(channels, channels, kernel_size, stride=1, padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1]).weight_norm(),
        ])
        self.convs.apply(init_weights)
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        for c in self.convs: x = c(self.act(x)) + x
        return x
    
    def remove_parametrizations(self):
        for c in self.convs: c.remove_weight_norm()

class ParallelResBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: tuple[int] = (3, 7, 11), dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),):
        super(ParallelResBlock, self).__init__()
        assert len(kernel_sizes) == len(dilation_sizes)
        self.blocks: list[ResBlock1] = nn.ModuleList([ResBlock1(channels, k, d,) for k, d in zip(kernel_sizes, dilation_sizes)])

    def forward(self, x): return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_parametrizations(self):
        for block in self.blocks: block.remove_parametrizations()
