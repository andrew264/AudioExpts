from typing import Optional, Tuple

import einops
import torch
from torch import Tensor, nn

def mask_sequence_tensor(tensor: Tensor, lengths: Tensor) -> Tensor:
    """
    For tensors containing sequences, zero out out-of-bound elements given lengths of every element in the batch.

    tensor: tensor of shape (B, L), (B, D, L) or (B, D1, D2, L),
    lengths: LongTensor of shape (B,)
    """
    batch_size, *_, max_lengths = tensor.shape

    if len(tensor.shape) == 2:
        mask = torch.ones(batch_size, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1')
    elif len(tensor.shape) == 3:
        mask = torch.ones(batch_size, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1')
    elif len(tensor.shape) == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1 1')
    else:
        raise ValueError('Can only mask tensors of shape B x L, B x D x L and B x D1 x D2 x L')

    return tensor * mask

def get_mask_from_lengths(lengths: Optional[Tensor] = None,x: Optional[Tensor] = None,) -> Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths: Optional[torch.tensor] (torch.tensor): 1D tensor with lengths
        x: Optional[torch.tensor] = tensor to be used on, last dimension is for mask
    Returns:
        mask (torch.tensor): num_sequences x max_length binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask

def snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    equation for snake activation function: x + (alpha + eps)^-1 * sin(alpha * x)^2
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake(nn.Module):
    """
    Snake activation function introduced in 'https://arxiv.org/abs/2006.08195'
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return snake(x, self.alpha)


class HalfSnake(nn.Module):
    """
    Activation which applies snake to the first half of input elements and leaky relu to the second half.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.snake_channels = channels // 2
        self.snake_act = Snake(self.snake_channels)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        snake_out = self.snake_act(x[:, : self.snake_channels, :])
        lrelu_out = self.lrelu(x[:, self.snake_channels :, :])
        out = torch.cat([snake_out, lrelu_out], dim=1)
        return out
    

class CodecActivation(nn.Module):
    """
    Choose between activation based on the input parameter.

    Args:
        activation: Name of activation to use. Valid options are "elu" (default), "lrelu", and "snake".
        channels: Input dimension.
    """

    def __init__(self, activation: str = "elu", channels: int = 1):
        super().__init__()
        activation = activation.lower()
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "snake":
            self.activation = Snake(channels)
        elif activation == "half_snake":
            self.activation = HalfSnake(channels)
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, x):
        return self.activation(x)
    
class ClampActivation(nn.Module):

    def __init__(self, min_value: float = -1.0, max_value: float = 1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.clamp(inputs, min=self.min_value, max=self.max_value)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_padding_2d(kernel_size: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    paddings = (get_padding(kernel_size[0], dilation[0]), get_padding(kernel_size[1], dilation[1]))
    return paddings


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding