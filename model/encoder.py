import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=384, num_layers=3):
        super(Encoder, self).__init__()

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
            ] + [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
                for _ in range(num_layers - 1)
            ]
        )

        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        self.relu = nn.ReLU()

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = self.relu(bn(conv(x)))
        return x.transpose(1, 2)
