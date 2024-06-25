import torch
from torch import nn
from torch.nn import functional as F
import lightning as L


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class VQVAEEncoder(nn.Module):
    def __init__(self, input_dim=(3, 64, 64), latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_channels = input_dim[0]

        self.layers = nn.Sequential(
            nn.Conv2d(self.num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 32),
            ResBlock(128, 32)
        )

        self.pre_quantized = nn.Conv2d(128, self.latent_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.layers(x)
        return self.pre_quantized(x)


class QuantizerEMA(nn.Module):
    def __init__(self, embedding_dim=16, num_embeddings=128, commitment_loss_factor=0.25, decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_factor = commitment_loss_factor
        self.decay = decay

        init_range = 1 / self.num_embeddings
        self.register_buffer("embeddings",
                             torch.empty(self.num_embeddings, self.embedding_dim).uniform_(-init_range, init_range))
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_embed",
                             torch.empty(self.num_embeddings, self.embedding_dim).uniform_(-init_range, init_range))

    def forward(self, z: torch.Tensor):
        z_flat = z.reshape(-1, self.embedding_dim)

        distances = torch.cdist(z_flat, self.embeddings, p=2.0)
        closest = distances.argmin(dim=1).unsqueeze(1)

        quantized_indices = closest.reshape(z.shape[:-1])
        one_hot = F.one_hot(closest, num_classes=self.num_embeddings).float().squeeze(1)

        quantized = torch.matmul(one_hot, self.embeddings).reshape_as(z)

        if self.training:
            self._update_embeddings(z_flat, one_hot)

        commitment_loss = F.mse_loss(quantized.detach(), z, reduction='sum')
        loss = self.commitment_loss_factor * commitment_loss

        quantized = z + (quantized - z).detach()
        return quantized.permute(0, 3, 1, 2), quantized_indices.unsqueeze(1), loss

    def _update_embeddings(self, z_flat, one_hot):
        n_i = one_hot.sum(dim=0)
        self.cluster_size.data = self.cluster_size * self.decay + n_i * (1 - self.decay)
        dw = torch.matmul(one_hot.t(), z_flat)
        self.ema_embed.data = self.ema_embed * self.decay + dw * (1 - self.decay)
        n = self.cluster_size.sum()
        cluster_size = ((self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)
        self.embeddings.data = self.ema_embed / cluster_size.unsqueeze(1)


class Quantizer(nn.Module):
    def __init__(self, embedding_dim=16, num_embeddings=128, commitment_loss_factor=0.25, quantization_loss_factor=1.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_factor = commitment_loss_factor
        self.quantization_loss_factor = quantization_loss_factor

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, z: torch.Tensor):
        distances = (
                (z.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
                + (self.embeddings.weight ** 2).sum(dim=-1)
                - 2 * z.reshape(-1, self.embedding_dim) @ self.embeddings.weight.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        quantized_indices = closest.reshape(z.shape[0], z.shape[1], z.shape[2])

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        quantized = one_hot_encoding @ self.embeddings.weight
        quantized = quantized.reshape_as(z)

        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="sum",
        )

        embedding_loss = F.mse_loss(
            quantized.reshape(-1, self.embedding_dim),
            z.detach().reshape(-1, self.embedding_dim),
            reduction="sum",
        )

        quantized = z + (quantized - z).detach()

        loss = (
                commitment_loss * self.commitment_loss_factor
                + embedding_loss * self.quantization_loss_factor
        )
        quantized = quantized.permute(0, 3, 1, 2)

        return quantized, quantized_indices.unsqueeze(1), loss


class VQVAEDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=(3, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_channels = output_dim[0]

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 128, kernel_size=1, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 32),
            ResBlock(128, 32),
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, self.num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class VQVAE(L.LightningModule):
    # https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/vq_vae/vq_vae_model.py
    def __init__(self, input_dim=(3, 64, 64), latent_dim=16, num_embeddings=128, commitment_loss_factor=0.25,
                 use_ema=False, quantization_loss_factor=1.0, decay=0.99):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VQVAEEncoder(input_dim, latent_dim)
        if use_ema:
            self.quantizer = QuantizerEMA(latent_dim, num_embeddings, commitment_loss_factor, decay)
        else:
            self.quantizer = Quantizer(latent_dim, num_embeddings, commitment_loss_factor, quantization_loss_factor)
        self.decoder = VQVAEDecoder(latent_dim, input_dim)

    @staticmethod
    def loss_fn(recon_x, x, vq_loss):
        bsz = x.size(0)
        recon_loss = F.mse_loss(recon_x.reshape(bsz, -1),
                                x.reshape(bsz, -1),
                                reduction='none').sum(dim=-1)

        return (
            (recon_loss + vq_loss).mean(dim=0),
            recon_loss.mean(dim=0),
            vq_loss.mean(dim=0),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1)
        quantized, quantized_indices, vq_loss = self.quantizer(z)
        recon_x = self.decoder(quantized)
        return recon_x, vq_loss, quantized

    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, loss, _ = self(x)
        loss, recon_loss, vq_loss = self.loss_fn(recon_x, x, loss)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_vq_loss", vq_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        recon_x, loss, _ = self(x)
        loss, recon_loss, vq_loss = self.loss_fn(recon_x, x, loss)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_vq_loss", vq_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
