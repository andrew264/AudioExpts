import itertools
from math import prod
from typing import Callable
from functools import partial

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor

from model.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from model.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from model.processors.melspec import MelSpectrogramFeatures
from model.layers.resblock import ParallelResBlock
from model.layers.conv1d import Conv1DNet, TransConv1DNet, init_weights


class HiFiGANGenerator(nn.Module):
    def __init__(
            self, hop_length: int = 512, num_mels: int = 512,
            upsample_rates: tuple[int] = (8, 8, 2, 2, 2), upsample_kernel_sizes: tuple[int] = (16, 16, 4, 4, 4),
            resblock_kernel_sizes: tuple[int] = (3, 7, 11), resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            upsample_initial_channel: int = 512, pre_conv_kernel_size: int = 13,
            post_conv_kernel_size: int = 13, activation: Callable[[Tensor], Tensor]=partial(F.silu, inplace=False)):
        super(HiFiGANGenerator, self).__init__()

        assert (prod(upsample_rates)==hop_length), f"hop_length must be {prod(upsample_rates)}"
        self.conv_pre = Conv1DNet(num_mels, upsample_initial_channel, pre_conv_kernel_size, stride=1).weight_norm()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.ups: list[TransConv1DNet] = nn.ModuleList([
            TransConv1DNet(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, stride=u,).weight_norm()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        ])
        self.ups.apply(init_weights)

        self.resblocks: list[ParallelResBlock] = nn.ModuleList()
        ch: int = None
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(ParallelResBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes,))

        self.act = activation
        self.conv_post = Conv1DNet(ch, 1, post_conv_kernel_size, stride=1).weight_norm()
    def forward(self, x):
        x = self.conv_pre(x)
        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            x = upsample_layer(self.act(x))
            if self.training: x = checkpoint(resblock_group, x, use_reentrant=False, )
            else: x = resblock_group(x)
        return torch.tanh(self.conv_post(self.act(x)))
    
    def remove_parametrizations(self):
        for up in self.ups: up.remove_weight_norm()
        for block in self.resblocks: block.remove_parametrizations()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

class HiFiGAN(L.LightningModule):
    def __init__(self, hop_length: int = 512, num_mels: int = 512,
            upsample_rates: tuple[int] = (8, 8, 2, 2, 2), upsample_kernel_sizes: tuple[int] = (16, 16, 4, 4, 4),
            resblock_kernel_sizes: tuple[int] = (3, 7, 11), resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            upsample_initial_channel: int = 512, pre_conv_kernel_size: int = 13,
            post_conv_kernel_size: int = 13, sample_rate=44100, n_fft=2048, win_length=2048, training: bool = True):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.melspec = MelSpectrogramFeatures(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, win_length=win_length, hop_length=hop_length)
        self.generator = HiFiGANGenerator(hop_length=hop_length, num_mels=num_mels, upsample_rates=upsample_rates, upsample_kernel_sizes=upsample_kernel_sizes,
                                          resblock_kernel_sizes=resblock_kernel_sizes, resblock_dilation_sizes=resblock_dilation_sizes,
                                          upsample_initial_channel=upsample_initial_channel, pre_conv_kernel_size=pre_conv_kernel_size, post_conv_kernel_size=post_conv_kernel_size)

        self.lr = 1e-4
        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()
        self.l1_factor = 45

        self.automatic_optimization = False
        self.train_discriminator = False

    def forward(self, x): return self.generator(x)

    def get_mel_spec(self, x: Tensor) -> Tensor: return self.melspec(x)

    def training_step(self, batch, batch_idx):
        audio_mel, audio = batch

        optim_g, optim_d = self.optimizers()

        # Discriminator step
        if self.train_discriminator:
            optim_d.zero_grad()
            with torch.no_grad():
                audio_pred = self(audio_mel)
            mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_pred)
            loss_disc_mpd, _, _ = self.discriminator_loss(disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen)
            msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_pred)
            loss_disc_msd, _, _ = self.discriminator_loss(disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen)
            loss_d = loss_disc_msd + loss_disc_mpd
            self.manual_backward(loss_d)
            optim_d.step()
        else: loss_d = 0.

        # Generator step
        optim_g.zero_grad()

        audio_pred = self(audio_mel)
        audio_pred_mel = self.get_mel_spec(audio_pred.squeeze(1))

        loss_mel = F.l1_loss(audio_pred_mel, audio_mel)
        if self.train_discriminator:
            _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_pred)
            _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_pred)
            loss_fm_mpd = self.feature_loss(fmap_mpd_real, fmap_mpd_gen)
            loss_fm_msd = self.feature_loss(fmap_msd_real, fmap_msd_gen)
            loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
            loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
        else:
            loss_gen_msd = loss_gen_mpd = loss_fm_msd = loss_fm_mpd = 0
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel * self.l1_factor
        self.manual_backward(loss_g)
        optim_g.step()

        self.update_lr()
        self.log_dict({'gen/total': loss_g, 'disc/total': loss_d}, prog_bar=True)

    def configure_optimizers(self):
        gen_params = self.generator.parameters()
        disc_params = itertools.chain(self.msd.parameters(), self.mpd.parameters())
        opt_g = torch.optim.AdamW(gen_params, lr=self.lr, betas=(0.8, 0.9), weight_decay=0.1)
        opt_d = torch.optim.AdamW(disc_params, lr=self.lr, betas=(0.8, 0.9), weight_decay=0.1)

        decay = .999
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, decay)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def update_lr(self,):
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()
