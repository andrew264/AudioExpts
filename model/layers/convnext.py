from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import Permute, StochasticDepth

from model.layers.conv1d import Conv1DNet

# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float=1e-6, data_format: Literal['channels_last', 'channels_first']="channels_last"):
        super().__init__()
        if data_format not in ["channels_last", "channels_first"]: raise NotImplementedError 
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last": return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6, mlp_ratio: float = 4.0, kernel_size: int = 7, dilation: int = 1,):
        super().__init__()

        self.block = nn.Sequential(
            Conv1DNet(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim),
            Permute([0, 2, 1]),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(in_features=dim, out_features=int(mlp_ratio * dim), bias=True),
            nn.GELU(),
            nn.Linear(in_features=int(mlp_ratio * dim), out_features=dim, bias=True),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim))) if layer_scale_init_value > 0 else None
        self.drop_path = StochasticDepth(drop_path, "row")

    def forward(self, x: Tensor, apply_residual: bool = True) -> Tensor:
        res = x
        x = self.block(x)
        if self.gamma is not None: x = self.gamma * x
        x = self.drop_path(x.permute(0, 2, 1))
        
        if apply_residual: x = res + x
        return x
    
class ConvNeXt1DEncoder(nn.Module):
    ### ConvNext without the classifier head
    def __init__(
        self, input_channels: int = 3, depths: list[int] = (3, 3, 9, 3), dims: list[int] = (96, 192, 384, 768), drop_path_rate: float = 0.0, layer_scale_init_value: float = 1e-6, kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(Conv1DNet(input_channels, dims[0], kernel_size=7,), LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),)
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"), nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),)
            self.downsample_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(*[ConvNeXt1DBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, kernel_size=kernel_size,) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x)