import os

import h5py
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader

from model import HiFiGAN
from model.processors.logmelspec import LogMelSpectrogram

feature_extractor = LogMelSpectrogram(sample_rate=44100, n_fft=2048, n_mels=160, hop_length=512, win_length=2048)
torch.set_float32_matmul_precision('medium')


class HDF5AudioDataset(Dataset):
    def __init__(self, hdf5_path, dtype = torch.float32):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.audio_group = self.hdf5_file['audio_data']
        self.keys = list(self.audio_group.keys())
        self.dtype = dtype
    
    def __len__(self): return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        audio_chunk = self.audio_group[key][:]
        audio_tensor = torch.tensor(audio_chunk, dtype=self.dtype)
        return audio_tensor[..., :64000]
    
    def close(self): self.hdf5_file.close()


def collate_fn(batch):
    waveforms = [*batch]
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    mel_specs = feature_extractor(waveforms)
    return mel_specs.to(dtype=torch.bfloat16), waveforms.to(dtype=torch.bfloat16)


def get_model():
    return HiFiGAN(sample_rate=44100, n_fft=2048, num_mels=160, hop_length=512, win_length=2048,
                   upsample_initial_channel=512, upsample_rates=(8,8,4,2), upsample_kernel_sizes=(16,16,4,4), training=True)


def dataloader():
    dl = DataLoader(HDF5AudioDataset('audio.h5', torch.bfloat16), batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    return dl


if __name__ == '__main__':
    model = get_model()
    if os.path.exists("weights/hifi_gan.pt"):
        model.generator.load_state_dict(torch.load("weights/hifi_gan.pt"))
        print('loaded weights')

    print(model)

    dl = dataloader()

    trainer = L.Trainer(precision="bf16-mixed", max_epochs=1)
    trainer.fit(model, dl)
    torch.save(model.generator.state_dict(), "weights/hifi_gan.pt")
