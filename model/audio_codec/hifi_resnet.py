from typing import Iterable
from torch import nn, Tensor

from model.audio_codec.blocks import Conv1dNorm
from model.audio_codec.utils import CodecActivation

class ResidualBlock(nn.Module):
    """
    The residual block structure defined by the HiFi-GAN V1 and V2 configurations.

    Args:
        channels: Input dimension.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        dilation: Dilation of the residual convolutions.
        dropout_rate: Dropout to apply to residuals.
        activation: Activation to apply in between residual convolutions.
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_rate: float = 0.0,
        activation: str = "lrelu",
    ):
        super(ResidualBlock, self).__init__()

        self.input_activation = CodecActivation(activation=activation, channels=channels)
        self.skip_activation = CodecActivation(activation=activation, channels=filters)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_conv = Conv1dNorm(
            in_channels=channels, out_channels=filters, kernel_size=kernel_size, dilation=dilation
        )
        self.skip_conv = Conv1dNorm(in_channels=filters, out_channels=channels, kernel_size=kernel_size)

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        conv_input = self.input_activation(inputs)
        skip_input = self.input_conv(inputs=conv_input, input_len=input_len)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(inputs=skip_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res
        return out


class HiFiGANResBlock(nn.Module):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple dilations.

    Args:
        channels: Input dimension.
        kernel_size: Kernel size of the residual blocks.
        dilations: List of dilations. One residual block will be created for each dilation in the list.
        activation: Activation for the residual blocks.
    """

    def __init__(self, channels: int, kernel_size: int, dilations: Iterable[int], activation: str):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    filters=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                )
                for dilation in dilations
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        out = inputs
        for res_block in self.res_blocks:
            out = res_block(inputs=out, input_len=input_len)
        return out


class HiFiGANResLayer(nn.Module):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple kernel sizes and dilations.
    One residual block is created for each combination of kernel size and dilation.

    Args:
        channels: Input dimension.
        kernel_sizes: List of kernel sizes.
        dilations: List of dilations.
        activation: Activation for the residual layers.

    """

    def __init__(self, channels: int, kernel_sizes: Iterable[int], dilations: Iterable[int], activation: str):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                HiFiGANResBlock(channels=channels, kernel_size=kernel_size, dilations=dilations, activation=activation)
                for kernel_size in kernel_sizes
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        residuals = [res_block(inputs=inputs, input_len=input_len) for res_block in self.res_blocks]
        out = sum(residuals) / len(residuals)
        return out