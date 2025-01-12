from typing import List, Tuple, Iterable

from einops import rearrange
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.audio_codec.blocks import Conv2dNorm

class PeriodDiscriminator(nn.Module):
    """
    Period discriminator introduced in HiFi-GAN https://arxiv.org/abs/2010.05646 which attempts to
    discriminate phase information by looking at equally spaced audio samples.

    Args:
        period: Spacing between audio sample inputs.
        lrelu_slope: Slope to use for activation. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss.
    """

    def __init__(self, period, lrelu_slope=0.1):
        super().__init__()
        self.period = period
        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(1, 32, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(32, 128, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(128, 512, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(512, 1024, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(1024, 1024, kernel_size=(5, 1), stride=(1, 1)),
            ]
        )
        self.conv_post = Conv2dNorm(1024, 1, kernel_size=(3, 1))

    def forward(self, audio: Tensor) -> Tuple[Tensor, List[Tensor]]:

        batch_size, time = audio.shape
        out = rearrange(audio, 'B T -> B 1 T')
        # Pad audio so that it is divisible by the period
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            out = F.pad(out, (0, n_pad), "reflect")
            time = time + n_pad
        # [batch, 1, (time / period), period]
        out = out.view(batch_size, 1, time // self.period, self.period)

        fmap = []
        for conv in self.conv_layers:
            # [batch, filters, (time / period / stride), period]
            out = torch.utils.checkpoint.checkpoint(conv, inputs=out, use_reentrant=False)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, (time / period / strides), period]
        score = self.conv_post(inputs=out)
        fmap.append(score)
        score = rearrange(score, "B 1 T C -> B C T")

        return score, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Wrapper class to aggregate results of multiple period discriminators.

    The periods are expected to be increasing prime numbers in order to maximize coverage and minimize overlap
    """

    def __init__(self, periods: Iterable[int] = (2, 3, 5, 7, 11), lrelu_slope=0.1):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=period, lrelu_slope=lrelu_slope) for period in periods]
        )

    def forward(self, audio_real: Tensor, audio_gen: Tensor) -> Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, fmap_real = discriminator(audio=audio_real)
            score_gen, fmap_gen = discriminator(audio=audio_gen)
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class DiscriminatorSTFT(nn.Module):
    """
    Discriminator network from EnCodec for Complex STFT input, but without dilations.

    Args:
        filters: number of filters to use in Conv2d layers
        lrelu_slope: Slope to use for activations. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss
    """

    def __init__(self, filters: int = 32, lrelu_slope: float = 0.1):
        super().__init__()

        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(2, filters, kernel_size=(3, 9)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 3)),
            ]
        )
        self.conv_post = Conv2dNorm(filters, 1, kernel_size=(3, 3))

    def forward(self, spec: Tensor) -> Tuple[Tensor, List[Tensor]]:
        fmap = []

        # [batch, 2, T_spec, fft]
        out = spec
        for conv in self.conv_layers:
            # [batch, filters, T_spec, fft // strides]
            out = torch.utils.checkpoint.checkpoint(conv, inputs=out, use_reentrant=False)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, T_spec, fft // 8]
        scores = self.conv_post(inputs=out)
        fmap.append(scores)
        scores = rearrange(scores, "B 1 T C -> B C T")

        return scores, fmap


class MultiBandDiscriminatorSTFT(nn.Module):
    """
    Multi-band STFT discriminator proposed in DAC (https://arxiv.org/abs/2306.06546).

    Computes the complex STFT for a given resolution and splits it into sub-bands,
    which are given to separate discriminator networks.

    Args:
        resolution: STFT resolution, provided as a tuple of 3 integers ordered (num_fft, hop_length, window_length)
        stft_bands: List of tuples, with each tuple having 2 float values (band_start, band_end).
            The floats are in the range [0, 1] representing the fraction of all stft bands.
            For example for n_fft=1024, the stft output has 513 dimensions.
            For band input [(0, 0.25), (0.25, 1.0)] it would use stft dimensions [0 through 127] and [128 through 512].
    """

    def __init__(self, resolution: Tuple[int], stft_bands: Iterable[Tuple[int]]):
        super().__init__()

        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.discriminators = nn.ModuleList([DiscriminatorSTFT() for _ in stft_bands])
        n_stft = self.n_fft // 2 + 1
        self.stft_bands = [(int(band[0] * n_stft), int(band[1] * n_stft)) for band in stft_bands]

    def compute_stft(self, audio: Tensor) -> Tensor:
        # [B, fft, T_spec]
        dtype = audio.dtype
        if dtype != torch.float16:
            audio = audio.to(dtype=torch.float16)
        fft = torch.stft(audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
        )
        fft = rearrange(fft, "B fft T -> B T fft")
        # [batch, 2, T_spec, fft]
        out = torch.stack([fft.real, fft.imag], dim=1)
        return out.to(dtype=dtype)

    def forward(self, audio: Tensor) -> Tuple[List[Tensor], List[List[Tensor]]]:
        scores_list = []
        fmap_list = []
        spec = self.compute_stft(audio)
        for band, disc in zip(self.stft_bands, self.discriminators):
            spec_band = spec[:, :, :, band[0] : band[1]]
            score, fmap = disc(spec=spec_band)
            scores_list.append(score)
            fmap_list.append(fmap)

        return scores_list, fmap_list


class MultiResolutionDiscriminatorSTFT(nn.Module):
    """
    Multi-resolution discriminator which creates a multi-band discriminator for each input resolution.

    Args:
        resolutions: List of STFT resolutions, each resolution provided as a tuple of 3 integers ordered
            (num_fft, hop_length, window_length)
        stft_bands: List of tuples, with each tuple having 2 float values (band_start, band_end).
            The floats are in the range [0, 1] representing the fraction of all stft bands.
            For example for n_fft=1024, the stft output has 513 dimensions.
            For band input [(0, 0.25), (0.25, 1.0)] it would use stft dimensions [0 through 127] and [128 through 512].
    """

    def __init__(self, resolutions: Iterable[Tuple[int]], stft_bands: Iterable[Tuple[int]]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [MultiBandDiscriminatorSTFT(resolution=resolution, stft_bands=stft_bands) for resolution in resolutions]
        )

    def forward(self, audio_real: Tensor, audio_gen: Tensor) -> Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []

        for disc in self.discriminators:
            score_real_i, fmap_real_i = disc(audio=audio_real)
            scores_real = scores_real + score_real_i
            fmaps_real = fmaps_real + fmap_real_i

            score_gen_i, fmap_gen_i = disc(audio=audio_gen)
            scores_gen = scores_gen + score_gen_i
            fmaps_gen = fmaps_gen + fmap_gen_i

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class Discriminator(nn.Module):
    """
    Wrapper class which takes a list of discriminators and aggregates the results across them.
    """

    def __init__(self, discriminators: Iterable[nn.Module]):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, audio_real: Tensor, audio_gen: Tensor) -> Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, score_gen, fmap_real, fmap_gen = discriminator(audio_real=audio_real, audio_gen=audio_gen)
            scores_real += score_real
            fmaps_real += fmap_real
            scores_gen += score_gen
            fmaps_gen += fmap_gen

        return scores_real, scores_gen, fmaps_real, fmaps_gen
