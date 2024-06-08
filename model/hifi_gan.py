import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size - 1) // 2 * d)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, padding=(kernel_size - 1) // 2)
            for _ in dilation
        ])

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = conv1(F.leaky_relu(x, 0.1))
            xt = conv2(F.leaky_relu(xt, 0.1))
            x = x + xt
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(self, in_channels=80, upsample_rates=None, upsample_kernel_sizes=None,
                 resblock_kernel_sizes=None, resblock_dilation_sizes=None):
        super(HiFiGANGenerator, self).__init__()

        upsample_initial_channel = 512

        self.upsample_rates = [8, 8, 2, 2] if upsample_rates is None else upsample_rates
        self.upsample_kernel_sizes = [16, 16, 4, 4] if upsample_kernel_sizes is None else upsample_kernel_sizes

        self.resblock_kernel_sizes = [3, 7, 11] if resblock_kernel_sizes is None else resblock_kernel_sizes
        self.num_kernel = len(self.resblock_kernel_sizes)

        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]\
            if resblock_dilation_sizes is None else resblock_dilation_sizes

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = []
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                               upsample_initial_channel // (2 ** (i + 1)),
                                               k,
                                               u,
                                               padding=(k - u) // 2))
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

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

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
