import lightning as L
import torch
import torch.nn.functional as F

from .melspec import MelSpectrogram
from .discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .hifi_gan import HiFiGANGenerator


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
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss = loss + (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0.
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += F.l1_loss(gl, rl)
        return loss * 2

    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0.
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    def training_step(self, batch, batch_idx):
        mels, wavs = batch
        wavs = wavs.unsqueeze(1)

        opt_g, opt_d = self.optimizers()

        # Discriminator step
        opt_d.zero_grad()
        y_g_hat = self(mels)
        y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

        # Period discriminator
        y_df_hat_r, y_df_hat_g, _, _ = self.period_discriminator(wavs, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(y_df_hat_r, y_df_hat_g)

        # Scale discriminator
        y_ds_hat_r, y_ds_hat_g, _, _ = self.scale_discriminator(wavs, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f

        loss_disc_all.backward()
        opt_d.step()
        self.log('train_loss_d', loss_disc_all, on_step=True, on_epoch=True, prog_bar=True)

        # Generator step
        opt_g.zero_grad()
        loss_mel = F.l1_loss(mels, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.period_discriminator(wavs, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.scale_discriminator(wavs, y_g_hat)
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        opt_g.step()

        self.log('train_loss_g', loss_gen_all, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.lr, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(
            list(self.period_discriminator.parameters()) + list(self.scale_discriminator.parameters()),
            lr=self.lr, betas=(0.8, 0.99)
        )
        return [opt_g, opt_d]
