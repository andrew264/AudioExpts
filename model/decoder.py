import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Decoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=384, output_dim=80, num_layers=6, num_head=8, dim_feedforward=1536):
        super(Decoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            activation=F.gelu,
            batch_first=True,
            bias=False
        )
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoded_features, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt = self.embedding(encoded_features)
        tgt = self.positional_encoding(tgt)

        output = self.transformer_decoder(
            tgt=tgt,
            memory=encoded_features,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        output = self.fc_out(output)
        return output
