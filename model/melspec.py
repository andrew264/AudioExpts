import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class MelSpectrogram(object):
    def __init__(self, n_mels=80):
        self.mel_basis = {}
        self.hann_window = {}
        self.n_fft = 1024
        self.num_mels = n_mels
        self.sampling_rate = 16000
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

        pad = int((self.n_fft - self.hop_size) / 2)
        y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=self.hann_window[str(y.device)], center=center,
                          pad_mode='reflect', normalized=False, onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis[str(self.fmax) + '_' + str(y.device)], spec)
        spec = dynamic_range_compression(spec)  # spectral normalize
        return spec

    def __call__(self, x, sr=None):
        out_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        return self.mel_spectrogram(x, sr).to(dtype=out_dtype)
