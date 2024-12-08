from typing import Tuple, Optional

from torch import nn, Tensor

from model.audio_codec.utils import get_padding, get_padding_2d, get_up_sample_padding, mask_sequence_tensor

class Conv1dNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: Optional[int] = None,):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)
        conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode="reflect",)
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    def remove_weight_norm(self):
        nn.utils.parametrize.remove_parametrizations(self.conv, "weight")

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvTranspose1dNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        conv = nn.ConvTranspose1d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, padding_mode="zeros",
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    def remove_weight_norm(self):
        nn.utils.parametrize.remove_parametrizations(self.conv, "weight")

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class Conv2dNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1),):
        super().__init__()
        assert len(kernel_size) == len(dilation)
        padding = get_padding_2d(kernel_size, dilation)
        conv = nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, dilation=dilation,
            padding=padding, padding_mode="reflect",
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    def remove_weight_norm(self):
        nn.utils.parametrize.remove_parametrizations(self.conv, "weight")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)
