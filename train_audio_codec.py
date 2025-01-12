import os
import h5py
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchtune.training import set_activation_checkpointing

from model.audio_codec.model import AudioCodecModel, load_config
from model.audio_codec.discriminators import Discriminator
from model.audio_codec.encoder import HiFiGANEncoder
from model.audio_codec.decoder import HiFiGANDecoder

torch.set_float32_matmul_precision('medium')

class HDF5AudioDataset(Dataset):
    def __init__(self, hdf5_path, dtype=torch.bfloat16):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.audio_group = self.hdf5_file['audio_data']
        self.keys = list(self.audio_group.keys())
        self.dtype = dtype

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        audio_chunk = self.audio_group[key][0]
        audio_tensor = torch.tensor(audio_chunk, dtype=self.dtype)
        audio_len = torch.tensor(audio_tensor.size(-1))
        return {'audio': audio_tensor, 'audio_lens': audio_len}

    def close(self): self.hdf5_file.close()

def train(model: L.LightningModule, dataloader: DataLoader):
    checkpoint_callback = ModelCheckpoint(dirpath="weights/audio_codec", filename="weights", save_weights_only=True, every_n_train_steps=1000)
    trainer = L.Trainer(max_epochs=1, precision="bf16-mixed", callbacks=[checkpoint_callback], max_steps=50000)
    trainer.fit(model, dataloader)
    sd = model.state_dict()
    torch.save(sd, "weights/audio_codec/weights.ckpt")

if __name__ == '__main__':
    cfg = load_config('weights/audio_codec/config1.yaml')
    model = AudioCodecModel(cfg=cfg).bfloat16().cuda()
    # model.forward = torch.compile(model.forward, mode='max-autotune')
    set_activation_checkpointing(model, auto_wrap_policy={HiFiGANEncoder, HiFiGANDecoder})
    ds = HDF5AudioDataset('audio.h5', dtype=torch.bfloat16, )
    dl = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True)
    # if os.path.exists('./weights/audio_codec.ckpt'):
    #     model.load_state_dict(torch.load('./weights/audio_codec.ckpt', weights_only=True,), strict=False)
    #     print('loaded weights')
    train(model, dl)
