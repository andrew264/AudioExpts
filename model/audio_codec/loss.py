from typing import List

from einops import rearrange
import torch
from torch import Tensor, nn

from model.audio_codec.features import FilterbankFeatures
from model.audio_codec.utils import get_mask_from_lengths, mask_sequence_tensor

Loss = torch.nn.modules.loss._Loss

class MaskedLoss(Loss):
    def __init__(self, loss_fn, loss_scale: float = 1.0):
        super(MaskedLoss, self).__init__()
        self.loss_scale = loss_scale
        self.loss_fn = loss_fn

    def forward(self, predicted: Tensor, target: Tensor, target_len: Tensor) -> Tensor:
        assert target.shape[2] == predicted.shape[2]

        # [B, D, T]
        loss = self.loss_fn(input=predicted, target=target)
        # [B, T]
        loss = torch.mean(loss, dim=1)
        # [B]
        loss = torch.sum(loss, dim=1) / torch.clamp(target_len, min=1.0)

        # [1]
        loss = torch.mean(loss)
        loss = self.loss_scale * loss

        return loss


class MaskedMAELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = nn.L1Loss(reduction='none')
        super(MaskedMAELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class MaskedMSELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = nn.MSELoss(reduction='none')
        super(MaskedMSELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class TimeDomainLoss(Loss):
    def __init__(self):
        super(TimeDomainLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()

    def forward(self, audio_real: Tensor, audio_gen: Tensor, audio_len: Tensor) -> Tensor:
        audio_real = rearrange(audio_real, "B T -> B 1 T")
        audio_gen = rearrange(audio_gen, "B T -> B 1 T")
        loss = self.loss_fn(target=audio_real, predicted=audio_gen, target_len=audio_len)
        return loss


class MultiResolutionMelLoss(Loss):
    """
    Multi-resolution log mel spectrogram loss.

    Args:
        sample_rate: Sample rate of audio.
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        mel_dims: Dimension of mel spectrogram to compute for each resolution. Should be same length as 'resolutions'.
        log_guard: Value to add to mel spectrogram to avoid taking log of 0.
    """

    def __init__(self, sample_rate: int, resolutions: List[List], mel_dims: List[int], log_guard: float = 1.0):
        super(MultiResolutionMelLoss, self).__init__()
        assert len(resolutions) == len(mel_dims)

        self.l1_loss_fn = MaskedMAELoss()
        self.l2_loss_fn = MaskedMSELoss()

        self.mel_features = torch.nn.ModuleList()
        for mel_dim, (n_fft, hop_len, win_len) in zip(mel_dims, resolutions):
            mel_feature = FilterbankFeatures(
                sample_rate=sample_rate,
                nfilt=mel_dim,
                n_window_size=win_len,
                n_window_stride=hop_len,
                n_fft=n_fft,
                pad_to=1,
                mag_power=1.0,
                log_zero_guard_type="add",
                log_zero_guard_value=log_guard,
                mel_norm=None,
                normalize=None,
                preemph=None,
                dither=0.0,
                use_grads=True,
            )
            self.mel_features.append(mel_feature)

    def forward(self, audio_real: Tensor, audio_gen: Tensor, audio_len: Tensor) -> tuple[Tensor, Tensor]:
        l1_loss = 0.0
        l2_loss = 0.0
        for mel_feature in self.mel_features:
            mel_real, mel_real_len = mel_feature(x=audio_real, seq_len=audio_len)
            mel_gen, _ = mel_feature(x=audio_gen, seq_len=audio_len)
            l1_loss += self.l1_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)
            l2_loss += self.l2_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)

        l1_loss /= len(self.mel_features)
        l2_loss /= len(self.mel_features)

        return l1_loss, l2_loss
    

class STFTLoss(Loss):
    """
    Log magnitude STFT loss.

    Args:
        resolution: Resolution of spectrogram, a list of 3 numbers ordered [num_fft, hop_length, window_length]
        log_guard: Value to add to magnitude spectrogram to avoid taking log of 0.
        sqrt_guard: Value to add to when computing absolute value of STFT to avoid NaN loss.
    """

    def __init__(self, resolution: List[int], log_guard: float = 1.0, sqrt_guard: float = 1e-5):
        super(STFTLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()
        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.log_guard = log_guard
        self.sqrt_guard = sqrt_guard

    def _compute_spectrogram(self, audio, spec_len):
        # [B, n_fft, T_spec]
        dtype = audio.dtype
        if dtype != torch.float16:
            audio = audio.to(dtype=torch.float16)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        # [B, n_fft, T_spec, 2]
        spec = torch.view_as_real(spec)
        # [B, n_fft, T_spec]
        spec_mag = torch.sqrt(spec.pow(2).sum(-1) + self.sqrt_guard)
        spec_log = torch.log(spec_mag + self.log_guard)
        spec_log = mask_sequence_tensor(spec_log, spec_len)
        return spec_log.to(dtype=dtype)

    def forward(self, audio_real: Tensor, audio_gen: Tensor, audio_len: Tensor) -> Tensor:
        spec_len = (audio_len // self.hop_length) + 1
        spec_real = self._compute_spectrogram(audio=audio_real, spec_len=spec_len)
        spec_gen = self._compute_spectrogram(audio=audio_gen, spec_len=spec_len)
        loss = self.loss_fn(predicted=spec_gen, target=spec_real, target_len=spec_len)
        return loss


class MultiResolutionSTFTLoss(Loss):
    """
    Multi-resolution log magnitude STFT loss.

    Args:
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        log_guard: Value to add to magnitude spectrogram to avoid taking log of 0.
        sqrt_guard: Value to add to when computing absolute value of STFT to avoid NaN loss.
    """

    def __init__(self, resolutions: List[List], log_guard: float = 1.0, sqrt_guard: float = 1e-5):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_fns = torch.nn.ModuleList(
            [STFTLoss(resolution=resolution, log_guard=log_guard, sqrt_guard=sqrt_guard) for resolution in resolutions]
        )

    def forward(self, audio_real: Tensor, audio_gen: Tensor, audio_len: Tensor) -> Tensor:
        loss = 0.0
        for loss_fn in self.loss_fns:
            loss += loss_fn(audio_real=audio_real, audio_gen=audio_gen, audio_len=audio_len)
        loss /= len(self.loss_fns)
        return loss


class SISDRLoss(Loss):
    """
    SI-SDR loss based off of torchmetrics.functional.audio.sdr.scale_invariant_signal_distortion_ratio
    with added support for masking.
    """

    def __init__(self, epsilon: float = 1e-8):
        super(SISDRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, audio_real: Tensor, audio_gen: Tensor, audio_len: Tensor) -> Tensor:
        mask = get_mask_from_lengths(x=audio_real, lengths=audio_len)
        audio_len = rearrange(audio_len, 'B -> B 1')

        # Shift audio to have zero-mean
        # [B, 1]
        target_mean = torch.sum(audio_real, dim=-1, keepdim=True) / audio_len
        pred_mean = torch.sum(audio_gen, dim=-1, keepdim=True) / audio_len

        # [B, T]
        target = audio_real - target_mean
        target = target * mask
        pred = audio_gen - pred_mean
        pred = pred * mask

        # [B, 1]
        ref_pred = torch.sum(pred * target, dim=-1, keepdim=True)
        ref_target = torch.sum(target**2, dim=-1, keepdim=True)
        alpha = (ref_pred + self.epsilon) / (ref_target + self.epsilon)

        # [B, T]
        target_scaled = alpha * target
        distortion = target_scaled - pred

        # [B]
        target_scaled_power = torch.sum(target_scaled**2, dim=-1)
        distortion_power = torch.sum(distortion**2, dim=-1)

        ratio = (target_scaled_power + self.epsilon) / (distortion_power + self.epsilon)
        si_sdr = 10 * torch.log10(ratio)

        # [1]
        loss = -torch.mean(si_sdr)
        return loss


class FeatureMatchingLoss(Loss):
    """
    Standard feature matching loss measuring the difference in the internal discriminator layer outputs
    (usually leaky relu activations) between real and generated audio, scaled down by the total number of
    discriminators and layers.
    """

    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()

    def forward(self, fmaps_real: List[List[Tensor]], fmaps_gen: List[List[Tensor]]) -> Tensor:
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
    
class RelativeFeatureMatchingLoss(Loss):
    """
    Relative feature matching loss as described in https://arxiv.org/pdf/2210.13438.pdf.

    This is similar to standard feature matching loss, but it scales the loss by the absolute value of
    each feature averaged across time. This might be slightly different from the paper which says the
    "mean is computed over all dimensions", which could imply taking the average across both time and
    features.

    Args:
        div_guard: Value to add when dividing by mean to avoid large/NaN values.
    """

    def __init__(self, div_guard=1e-3):
        super(RelativeFeatureMatchingLoss, self).__init__()
        self.div_guard = div_guard

    def forward(self, fmaps_real: List[List[Tensor]], fmaps_gen: List[List[Tensor]]) -> Tensor:
        loss = 0.0
        for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
            # [B, ..., time]
            for feat_real, feat_gen in zip(fmap_real, fmap_gen):
                # [B, ...]
                feat_mean = torch.mean(torch.abs(feat_real), dim=-1)
                diff = torch.mean(torch.abs(feat_real - feat_gen), dim=-1)
                feat_loss = diff / (feat_mean + self.div_guard)
                # [1]
                feat_loss = torch.mean(feat_loss) / len(fmap_real)
                loss += feat_loss

        loss /= len(fmaps_real)

        return loss

class GeneratorSquaredLoss(Loss):
    def forward(self, disc_scores_gen: List[Tensor]) -> Tensor:
        loss = 0.0
        for disc_score_gen in disc_scores_gen:
            loss += torch.mean((1 - disc_score_gen) ** 2)

        loss /= len(disc_scores_gen)

        return loss
    
class DiscriminatorSquaredLoss(Loss):
    def forward(self, disc_scores_real: List[Tensor], disc_scores_gen: List[Tensor]) -> Tensor:
        loss = 0.0
        for disc_score_real, disc_score_gen in zip(disc_scores_real, disc_scores_gen):
            loss_real = torch.mean((1 - disc_score_real) ** 2)
            loss_gen = torch.mean(disc_score_gen**2)
            loss += (loss_real + loss_gen) / 2

        loss /= len(disc_scores_real)

        return loss
    
 