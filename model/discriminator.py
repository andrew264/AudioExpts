from typing import Tuple, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Conv2d, Conv1d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

def get_padding(kernel_size, dilation=1): return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [32, 128, 512, 1024]
        self.convs = nn.ModuleList([norm_f(Conv2d(1, conv_ch[0], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(conv_ch[0], conv_ch[1], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(conv_ch[1], conv_ch[2], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(conv_ch[2], conv_ch[3], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(conv_ch[3], conv_ch[3], (kernel_size, 1), 1, padding=(2, 0))), ])
        self.conv_post = norm_f(Conv2d(conv_ch[3], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super(DiscriminatorS, self).__init__()
        conv_ch = [128, 256, 512, 1024]
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [norm_f(Conv1d(1, conv_ch[0], 15, 1, padding=7)), norm_f(Conv1d(conv_ch[0], conv_ch[0], 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(conv_ch[0], conv_ch[1], 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[1], conv_ch[2], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[2], conv_ch[3], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 41, 1, groups=16, padding=20)), norm_f(Conv1d(conv_ch[3], conv_ch[3], 5, 1, padding=2)), ])
        self.conv_post = norm_f(nn.Conv1d(conv_ch[3], 1, 3, 1, padding=1))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
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
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
