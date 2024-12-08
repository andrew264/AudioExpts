from typing import Iterable, Tuple

from torch import nn, Tensor
from einops import rearrange

from model.audio_codec.blocks import Conv1dNorm, ConvTranspose1dNorm
from model.audio_codec.hifi_resnet import HiFiGANResLayer
from model.audio_codec.utils import CodecActivation, ClampActivation


class HiFiGANDecoder(nn.Module):
    """
    Codec decoder using the HiFi-GAN generator architecture.

    Default parameters match the HiFi-GAN V1 configuration for 22.05khz.

    Args:
        input_dim: Input dimension.
        up_sample_rates: Rate to upsample for each decoder block. The product of the upsample rates should be the same
            as the overall downsample rate for your encoder. For example, a symmetric encoder/decoder can be created
            with encoder downsample rates [2, 2, 8, 8] and decoder upsample rates [8, 8, 2, 2].
        base_channels: Number of filters in the first convolution. The number of channels will be cut in
            half after each upsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and upsample layers, defaults to leaky relu.
        output_activation: Activation to apply to output. To produce a valid audio signal, it should output values in
         the range [-1.0, 1.0]. Supports "tanh" and "clamp".
    """

    def __init__(
        self,
        input_dim: int,
        up_sample_rates: Iterable[int] = (8, 8, 2, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        output_activation: str = "tanh",
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=input_dim, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=up_sample_rate
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
            )
            self.res_layers.append(res_layer)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size)
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()
        else:
            raise ValueError(f"Invalid audio output activation {output_activation}")

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tuple[Tensor, Tensor]:
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
        ):
            audio_len = audio_len * up_sample_rate
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len