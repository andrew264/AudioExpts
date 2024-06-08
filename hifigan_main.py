import os

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from model import HiFiGAN, MelSpectrogram

feature_extractor = MelSpectrogram()
torch.set_float32_matmul_precision('high')


class LibreSpeechDataset(Dataset):
    def __init__(self, split: str = "train.360", streaming: bool = False):
        super().__init__()
        self._data = load_dataset("/home/andrew264/datasets/librispeech_asr", "clean",
                                  split=split, streaming=streaming, trust_remote_code=True)
        self.seqment_size = 16384

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        item = self._data[item]
        audio = torch.from_numpy(item['audio']['array']).to(dtype=torch.float32)
        if audio.size(0) < self.seqment_size:
            audio = torch.cat([audio, torch.zeros(self.seqment_size - audio.size(0))])
        else:
            audio = audio[:self.seqment_size]
        return audio


def collate_fn(batch):
    waveforms = [*batch]
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(dtype=torch.bfloat16)
    mel_specs = feature_extractor(waveforms)
    return mel_specs, waveforms


def get_model():
    return HiFiGAN()


def dataloader():
    dl = DataLoader(LibreSpeechDataset(), batch_size=8, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    return dl


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    model = get_model()
    if os.path.exists("generator.pth"):
        model.generator.load_state_dict(torch.load("generator.pth"))

    print(model)

    dl = dataloader()

    trainer = L.Trainer(precision="bf16-true", max_epochs=3)
    trainer.fit(model, dl)
    torch.save(model.generator.state_dict(), "generator.pth")
