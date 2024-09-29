import math
from typing import Self
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

def remove_padding_1d(x: Tensor, paddings: tuple[int, int]) -> Tensor:
    """Remove padding from a 1D tensor."""
    left_pad, right_pad = paddings
    assert 0 <= left_pad <= x.shape[-1] and 0 <= right_pad <= x.shape[-1], (left_pad, right_pad)
    return x[..., left_pad: x.shape[-1] - right_pad]

def calculate_extra_padding(x: Tensor, kernel_size: int, stride: int, padding: int = 0) -> int:
    """Calculate extra padding required to maintain output shape after 1D convolution."""
    input_length = x.shape[-1]
    num_frames = (input_length - kernel_size + padding) / stride + 1
    ideal_length = (math.ceil(num_frames) - 1) * stride + (kernel_size - padding)
    return ideal_length - input_length

def pad_tensor_1d(x: Tensor, paddings: tuple[int, int], mode: str = "constant", value: float = 0.0) -> Tensor:
    """Pad a 1D tensor with specified mode and value."""
    left_pad, right_pad = paddings
    assert left_pad >= 0 and right_pad >= 0, (left_pad, right_pad)

    if mode == "reflect":
        max_pad = max(left_pad, right_pad)
        extra_pad = max(0, max_pad - x.shape[-1] + 1)
        x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        return padded[..., :padded.shape[-1] - extra_pad]
    return F.pad(x, paddings, mode, value)

def init_weights(m: nn.Module, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d): m.weight.data.normal_(mean, std)

class Conv1DNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1):
        super(Conv1DNet, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.stride = stride
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1

    def forward(self, x: Tensor) -> Tensor:
        padding = self.effective_kernel_size - self.stride
        extra_padding = calculate_extra_padding(x, self.effective_kernel_size, self.stride, padding)
        x = pad_tensor_1d(x, (padding, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()
    
    def weight_norm(self, name: str="weight", dim=0) -> Self:
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self) -> Self:
        self.conv = remove_parametrizations(self.conv, 'weight')
        return self

class TransConv1DNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, dilation=1):
        super(TransConv1DNet, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = remove_padding_1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name: str="weight", dim=0) -> Self:
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self) -> Self:
        self.conv = remove_parametrizations(self.conv, 'weight')
        return self
