from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss as Loss

from model.processors.melspec import MelSpectrogramFeatures

class FeatureMatchingLoss(Loss):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
    def forward(self, fmaps_real: List[Tensor], fmaps_gen: List[Tensor]) -> Tensor:
        loss = 0.0
        for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
            # [B, ..., time]
            for feat_real, feat_gen in zip(fmap_real, fmap_gen):
                # [B, ...]
                diff = torch.abs(feat_real - feat_gen)
                feat_loss = torch.mean(diff) / len(fmap_real)
                loss += feat_loss
        loss /= len(fmaps_real)
        return loss
    
class DiscriminatorLoss(Loss):
    def forward(self, disc_real_outputs: List[Tensor], disc_generated_outputs: List[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses
    
class GeneratorLoss(Loss):
    def forward(self, disc_outputs: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
    
class MelSpecReconstructionLoss(Loss):
    def __init__(self, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, padding="center",):
        super().__init__()
        self.mel_spec = MelSpectrogramFeatures(sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, padding=padding)
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        mel_hat = self.mel_spec(y_hat)
        mel = self.mel_spec(y)
        return self.loss_fn(mel, mel_hat)