import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram, Spectrogram


class MelSpectrogramFeatures(nn.Module):
    def __init__(self, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, padding="center"):
        super().__init__()
        self.mel_spec = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                       f_min=0, f_max=None, n_mels=n_mels, center=padding == "center",
                                       power=1, normalized=False,)
        self.padding = padding
        self.hop_length = hop_length
        self.sample_rate = sr

    def forward(self, audio: Tensor):
        out_dtype = audio.dtype
        if audio.dtype != torch.float32: audio = audio.float()
        if self.padding == "same":
            pad = (self.mel_spec.n_fft - self.mel_spec.hop_length) // 2
            audio = F.pad(audio, (pad, pad), mode="reflect")
        features = self.mel_spec(audio)
        features = features.squeeze(1)[..., 1:]  # without this there will be problems
        features = self.compress(features)
        return features.to(dtype=out_dtype)
    def compress(self, x: Tensor) -> Tensor: return torch.log(torch.clamp(x, min=1e-5))
    def decompress(self, x: Tensor) -> Tensor: return torch.exp(x)


class SpectrogramFeatures(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, center=True, pad_mode='reflect'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center, pad_mode=pad_mode)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        if dtype != torch.float32: x = x.float()
        return self.spec(x).to(dtype=dtype)
