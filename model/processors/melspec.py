import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn


def safe_log(x, clip_val=1e-7):
    return torch.log(torch.clip(x, min=clip_val))


class MelSpectrogram(object):
    def __init__(self, n_mels=80, sr=16000):
        self.mel_basis = {}
        self.hann_window = {}
        self.n_fft = 1024
        self.num_mels = n_mels
        self.sampling_rate = sr
        self.hop_size = 256
        self.win_size = 1024
        self.fmin = 0.
        self.fmax = 8000.

    def mel_spectrogram(self, y, sr=None, center=False):
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
        if sr is None:
            sr = self.sampling_rate

        fmax_key = f'{self.fmax}_{y.device}'
        if fmax_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sr, n_fft=self.n_fft, n_mels=self.num_mels,
                                 fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[fmax_key] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.win_size).to(y.device)

        pad = (self.n_fft - self.hop_size) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=self.hann_window[str(y.device)], center=center,
                          pad_mode='reflect', normalized=False, onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis[str(self.fmax) + '_' + str(y.device)], spec)
        spec = safe_log(spec)
        return spec

    def __call__(self, x, sr=None):
        out_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        return self.mel_spectrogram(x, sr).to(dtype=out_dtype)


class MelSpectrogram2(object):
    def __init__(self, sr=16000, n_fft=1024, hop_length=256,
                 win_length=1024, n_mels=80, padding="center", device='cuda'):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=0,
            f_max=8000,
            n_mels=n_mels,
            center=padding == "center",
            power=2,
            normalized=False,
        ).to(device)
        self.device = device
        self.padding = padding

    def __call__(self, audio: torch.Tensor):
        out_dtype = audio.dtype
        out_device = audio.device
        if audio.dtype != torch.float32:
            audio = audio.to(dtype=torch.float32)
        audio = audio.to(device=self.device)
        if self.padding == "same":
            pad = (self.mel_spec.n_fft - self.mel_spec.hop_length) // 2
            audio = F.pad(audio, (pad, pad), mode="reflect")
        features = self.mel_spec(audio)
        features = safe_log(features)
        return features.to(dtype=out_dtype, device=out_device)


class Spectrogram(object):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center, pad_mode=pad_mode
        )

    def __call__(self, x):
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        return self.spec(x).to(dtype=dtype)
