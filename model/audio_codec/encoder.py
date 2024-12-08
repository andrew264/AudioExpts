from typing import Iterable, Tuple

from torch import nn, Tensor
from einops import rearrange

from model.audio_codec.blocks import Conv1dNorm
from model.audio_codec.hifi_resnet import HiFiGANResLayer
from model.audio_codec.utils import CodecActivation, get_down_sample_padding


class HiFiGANEncoder(nn.Module):
    """
    Audio encoder created by inverting the HiFi-GAN decoder.

    Args:
        encoded_dim: Dimension of encoder output.
        down_sample_rates: Rate to upsample for each decoder block. The product of the downsample rates will
            determine the output token rate. For example 2 * 2 * 8 * 8 = 256 samples per token.
        base_channels: Number of filters in the first convolution. The number of channels will be doubled after each
            downsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and downsample layers, defaults to leaky relu.
    """

    def __init__(
        self,
        encoded_dim: int,
        down_sample_rates: Iterable[int] = (2, 2, 8, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for down_sample_rate in self.down_sample_rates:
            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
            )
            self.res_layers.append(res_layer)

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate

            padding = get_down_sample_padding(kernel_size=kernel_size, stride=down_sample_rate)
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=padding,
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size)

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    def forward(self, audio: Tensor, audio_len: Tensor) -> Tuple[Tensor, Tensor]:
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for act, res_layer, down_sample_conv, down_sample_rate in zip(
            self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_layer(inputs=out, input_len=encoded_len)
            out = act(out)

            encoded_len = encoded_len // down_sample_rate
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len
