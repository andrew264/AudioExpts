import torch
import torchaudio.functional as F
from torch import Tensor, nn


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, win_length=2048, hop_length=512, n_mels=128, center=False, f_min=0.0, f_max=None, mode="pow2_sqrt",):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)
        self.mode = mode

        self.window: Tensor
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        fb = F.melscale_fbanks(n_freqs=self.n_fft // 2 + 1,
                               f_min=self.f_min,
                               f_max=self.f_max,
                               n_mels=self.n_mels,
                               sample_rate=self.sample_rate,
                               norm="slaney",
                               mel_scale="slaney",)
        self.fb: Tensor
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, x: Tensor, return_linear: bool = False, sample_rate: int = None) -> Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        spec = self._compute_spectrogram(x)

        mel_spec = self.apply_mel_scale(spec)
        mel_spec = self.compress(mel_spec)

        if return_linear: return mel_spec, self.compress(spec)
        return mel_spec

    def _compute_spectrogram(self, x: Tensor) -> Tensor:
        if x.ndim == 3: x = x.squeeze(1)

        x = torch.nn.functional.pad(x.unsqueeze(1),
                                    ((self.win_length - self.hop_length) // 2,
                                     (self.win_length - self.hop_length + 1) // 2,),
                                    mode="reflect",).squeeze(1)

        spec = torch.stft(x, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,)

        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt": spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        return spec

    def apply_mel_scale(self, x: Tensor) -> Tensor: return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)
    def compress(self, x: Tensor) -> Tensor: return torch.log(torch.clamp(x, min=1e-5))
    def decompress(self, x: Tensor) -> Tensor: return torch.exp(x)
    