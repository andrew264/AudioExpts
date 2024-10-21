from math import prod
from typing import Any, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L

from model.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from model.loss import MelSpecReconstructionLoss, FeatureMatchingLoss, GeneratorLoss, DiscriminatorLoss
from model.layers.convnext import ConvNeXt1DEncoder
from model.layers.fsq import FiniteScalarQuantize
from model.hifi_gan import HiFiGANGenerator
from model.processors.logmelspec import LogMelSpectrogram
from model.processors.melspec import MelSpectrogramFeatures

class Cfg:
    # spec
    sample_rate: int = 44100
    n_mels: int = 160
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    # backbone
    input_channels: int = 160
    depths: tuple[int] = (3, 3, 9, 3)
    dims: tuple[int] = (128, 256, 384, 512)
    drop_path_rate: float = .2
    kernel_size: int = 7
    # head
    upsample_rates: tuple[int] = (8, 8, 2, 2, 2)
    upsample_kernel_sizes: tuple[int] = (16, 16, 4, 4, 4)
    resblock_kernel_sizes: tuple[int] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    head_n_mels: int = 512
    upsample_initial_channel: int = 512
    pre_conv_kernel_size: int = 13
    post_conv_kernel_size: int = 13
    # quantizer
    input_dims: int = 512
    n_groups: int = 8
    n_codebooks: int = 1
    levels: tuple[int] = (8, 5, 5, 5)
    downsample_factors: tuple[int] = (2, 2)

def exists(x: Optional[Any]) -> bool: return x is not None

def sequence_mask(length, max_length=None):
    if not exists(max_length): max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class AudioVQGAN(L.LightningModule):
    def __init__(self, cfg: Cfg, lr: float = 1e-5, dtype=torch.bfloat16):
        super().__init__()
        # self.feature_extractor = MelSpectrogramFeatures(sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
        #                                                 n_mels=cfg.n_mels)
        self.spec_transform = LogMelSpectrogram(sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
                                                hop_length=cfg.hop_length, n_mels=cfg.n_mels,)
        self.backbone = ConvNeXt1DEncoder(input_channels=cfg.input_channels, depths=cfg.depths, dims=cfg.dims, drop_path_rate=cfg.drop_path_rate,
                                          kernel_size=cfg.kernel_size).to(dtype=dtype)
        self.quantizer = FiniteScalarQuantize(input_dim=cfg.input_dims, n_codebooks=cfg.n_codebooks, n_groups=cfg.n_groups, levels=cfg.levels,
                                              downsample_factor=cfg.downsample_factors).to(dtype=dtype)
        self.head = HiFiGANGenerator(hop_length=cfg.hop_length, num_mels=cfg.head_n_mels, upsample_rates=cfg.upsample_rates,
                                     upsample_kernel_sizes=cfg.upsample_kernel_sizes, resblock_kernel_sizes=cfg.resblock_kernel_sizes,
                                     resblock_dilation_sizes=cfg.resblock_dilation_sizes, upsample_initial_channel=cfg.upsample_initial_channel,
                                     pre_conv_kernel_size=cfg.pre_conv_kernel_size, post_conv_kernel_size=cfg.post_conv_kernel_size).to(dtype=dtype)
        self.lr = lr
        self.downsample_factor = prod(cfg.downsample_factors)

        self.automatic_optimization = False

        # discriminators
        self.mpd = MultiPeriodDiscriminator().to(dtype=dtype)
        self.msd = MultiScaleDiscriminator().to(dtype=dtype)

        # losses
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feature_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length, n_mels=cfg.n_mels)
        self.mel_loss_coeff = 2.

        self.train_discriminator = False

    def configure_optimizers(self):
        disc_params = [dict(params=self.mpd.parameters()), dict(params=self.msd.parameters())]
        gen_params = [dict(params=self.spec_transform.parameters()), dict(params=self.backbone.parameters()),
                      dict(params=self.quantizer.parameters())]

        opt_d = torch.optim.AdamW(disc_params, lr=self.lr, betas=(0.8, 0.9))
        opt_g = torch.optim.AdamW(gen_params, lr=self.lr, betas=(0.8, 0.9))

        decay = .999
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, decay)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, **kwargs):
        audio = batch
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # discriminator
        if self.train_discriminator:
            opt_d.zero_grad()
            with torch.no_grad():
                audio_hat, _ = self(audio)

            mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_hat)
            loss_mpd, loss_mpd_real, _ = self.disc_loss(disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen)
            loss_mpd /= len(loss_mpd_real)

            msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_hat)
            loss_msd, loss_msd_real, _ = self.disc_loss(disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen)
            loss_msd /= len(loss_msd_real)

            loss_d = loss_msd + loss_mpd
            self.manual_backward(loss_d)
            opt_d.step()
            sch_d.step()
            self.log('disc/total', loss_d, prog_bar=True)
            self.log('disc/mpd', loss_mpd)
            self.log('disc/msd', loss_msd)

        # generator
        audio_hat, quant_loss = self(audio)
        if self.train_discriminator:
            _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_hat)
            loss_gen_mpd, list_loss_gen_mpd = self.gen_loss(disc_outputs=mpd_score_gen)
            loss_gen_mpd /= len(list_loss_gen_mpd)

            _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_hat)
            loss_gen_msd, list_loss_gen_msd = self.gen_loss(disc_outputs=msd_score_gen)
            loss_gen_msd /= len(list_loss_gen_msd)

            loss_fm_mpd = self.feature_loss(fmap_mpd_real, fmap_mpd_gen)
            loss_fm_msd = self.feature_loss(fmap_msd_real, fmap_msd_gen)
            self.log('gen/mpd', loss_gen_mpd)
            self.log('gen/msd', loss_gen_msd)
            self.log('gen/feature_mpd', loss_fm_mpd)
            self.log('gen/feature_msd', loss_fm_msd)
        else:
            loss_gen_msd = loss_gen_mpd = loss_fm_msd = loss_fm_mpd = 0
        mel_loss = self.melspec_loss(audio_hat, audio)

        loss = (loss_gen_mpd + loss_gen_msd + loss_fm_mpd + loss_fm_msd + self.mel_loss_coeff * mel_loss + quant_loss * .5)

        self.manual_backward(loss)
        opt_g.step()
        sch_g.step()
        self.log('gen/total', loss, prog_bar=True)
        self.log("gen/mel_loss", mel_loss)
        self.log("gen/quant_loss", quant_loss)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.spec_transform(x)

        x = self.backbone(x)
        if exists(mask): x = x * mask

        vq_result = self.quantizer(x)
        quant_loss = 0
        if self.training:
            quant_loss = F.l1_loss(vq_result.z, x)
        x = vq_result.z

        if exists(mask): x = x * mask

        x = self.head(x)
        if x.ndim == 2: x = x[:, None, :]
        return x, quant_loss

    def encode(self, audios: Tensor, audio_lengths: Tensor) -> Tensor:
        mels = self.spec_transform(audios)
        mel_lengths = audio_lengths // self.spec_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :]
        mels = mels * mel_masks_float_conv

        encoded_features = self.backbone(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // self.downsample_factor
        encoded = self.quantizer.encode(encoded_features)

        return encoded, feature_lengths

    def decode(self, indices, feature_lengths) -> torch.Tensor:
        mel_masks = sequence_mask(feature_lengths * self.downsample_factor, indices.shape[2] * self.downsample_factor, )
        mel_masks_float_conv = mel_masks[:, None, :]
        audio_lengths = feature_lengths * self.downsample_factor * self.spec_transform.hop_length
        audio_masks = sequence_mask(audio_lengths, indices.shape[2] * self.downsample_factor * self.spec_transform.hop_length, )
        audio_masks_float_conv = audio_masks[:, None, :]

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        x = self.head(z) * audio_masks_float_conv

        return x, audio_lengths

    def remove_parametrizations(self):
        if hasattr(self.head, "remove_parametrizations"):
            self.head.remove_parametrizations()
