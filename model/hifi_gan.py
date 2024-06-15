from typing import Tuple, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .melspec import MelSpectrogram


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int,
                 kernel_size: int = 5,
                 stride: int = 3, ):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, conv in enumerate(self.convs):
            x = F.leaky_relu(conv(x), 0.1)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return x.view(b, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2) for _ in range(2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size - 1) // 2 * d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, padding=(kernel_size - 1) // 2))
            for _ in dilation
        ])

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = conv1(F.leaky_relu(x, 0.1))
            xt = conv2(F.leaky_relu(xt, 0.1))
            x = x + xt
        return x


class ResidualBlock2(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size - 1) // 2 * d))
            for d in dilation
        ])

    def forward(self, x):
        for conv in self.convs:
            xt = conv(F.leaky_relu(x, 0.1))
            x = x + xt
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(self, in_channels=80, upsample_rates=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4),
                 resblock_kernel_sizes=(3, 7, 11), resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super(HiFiGANGenerator, self).__init__()

        upsample_initial_channel = 512

        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.num_kernel = len(self.resblock_kernel_sizes)
        self.resblock_dilation_sizes = resblock_dilation_sizes

        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = []
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                                           upsample_initial_channel // (2 ** (i + 1)),
                                                           k,
                                                           u,
                                                           padding=(k - u) // 2)))
        self.ups = nn.Sequential(*self.ups)

        self.resblocks = []
        for i in range(len(self.ups)):
            resblock_list = []

            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes,
                                           self.resblock_dilation_sizes)):
                resblock_list.append(ResidualBlock(ch, k, d))
            resblock_list = nn.Sequential(*resblock_list)
            self.resblocks.append(resblock_list)
        self.resblocks = nn.Sequential(*self.resblocks)

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)

        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = upsample_layer(x)
            xs = 0
            for resblock in resblock_group:
                xs += resblock(x)
            x = xs / self.num_kernel
        x = F.leaky_relu(x)

        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class Denoiser(nn.Module):
    def __init__(self, hifigan, filter_length=1024, hop_size=256, win_length=1024):
        super(Denoiser, self).__init__()
        self.filter_length = filter_length
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length)
        self.register_buffer('bias_spec', self.compute_bias_spec(hifigan), persistent=False)

    def compute_bias_spec(self, hifigan):
        with torch.no_grad():
            mel_input = torch.zeros(1, 80, 100)
            bias_audio = hifigan(mel_input).float().squeeze()
            bias_spec = torch.stft(bias_audio, self.filter_length, self.hop_size, self.win_length,
                                   window=self.window, return_complex=True)[:, 0][:, None]
        return bias_spec

    def forward(self, audio, strength=0.05):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio_stft = torch.stft(audio, self.filter_length, self.hop_size, self.win_length,
                                window=self.window, return_complex=True)

        audio_spec_denoised = audio_stft - self.bias_spec * strength
        audio_denoised = torch.istft(audio_spec_denoised, self.filter_length, self.hop_size,
                                     self.win_length, window=self.window, return_complex=False)
        return audio_denoised


class HiFiGAN(L.LightningModule):
    def __init__(self, training: bool = True):
        super().__init__()
        if training:
            self.period_discriminator = MultiPeriodDiscriminator()
            self.scale_discriminator = MultiScaleDiscriminator()
            self.melspec = MelSpectrogram()
        self.generator = HiFiGANGenerator()
        self.lr = 2e-4
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = torch.tensor(0., device=disc_real_outputs[0].device)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss = loss + (r_loss + g_loss)
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses

    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = torch.tensor(0., device=fmap_r[0][0].device)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss

    @staticmethod
    def generator_loss(disc_outputs):
        loss = torch.tensor(0., device=disc_outputs[0].device)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    def training_step(self, batch, batch_idx):
        mels, wavs = batch
        wavs = wavs.unsqueeze(1)

        opt_g, opt_d = self.optimizers()

        # Discriminator step
        y_g_hat = self(mels)
        y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

        # Period discriminator
        y_df_hat_r, y_df_hat_g, _, _ = self.period_discriminator(wavs, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(y_df_hat_r, y_df_hat_g)

        # Scale discriminator
        y_ds_hat_r, y_ds_hat_g, _, _ = self.scale_discriminator(wavs, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f

        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)
        opt_d.step()

        # Generator step
        loss_mel = F.l1_loss(mels, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.period_discriminator(wavs, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.scale_discriminator(wavs, y_g_hat)
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g) / len(fmap_f_r)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g) / len(fmap_s_r)
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_f = loss_gen_f / len(losses_gen_f)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
        loss_gen_s = loss_gen_s / len(losses_gen_s)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        opt_g.step()

        self.log_dict({
            'train_loss_g': loss_gen_all,
            'train_loss_d': loss_disc_all,
        }, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.lr, betas=(0.8, 0.9))
        opt_d = torch.optim.AdamW(
            list(self.period_discriminator.parameters()) + list(self.scale_discriminator.parameters()),
            lr=self.lr, betas=(0.8, 0.9)
        )
        return [opt_g, opt_d]
