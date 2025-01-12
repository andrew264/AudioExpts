from typing import Optional, List, Tuple, Literal
import itertools

import lightning as L
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
import yaml

from model.audio_codec.decoder import HiFiGANDecoder
from model.audio_codec.discriminators import Discriminator, MultiPeriodDiscriminator, MultiResolutionDiscriminatorSTFT
from model.audio_codec.encoder import HiFiGANEncoder
from model.audio_codec.loss import DiscriminatorSquaredLoss, FeatureMatchingLoss, GeneratorSquaredLoss, MultiResolutionMelLoss, MultiResolutionSTFTLoss, RelativeFeatureMatchingLoss, TimeDomainLoss, SISDRLoss
from model.audio_codec.quantizer import GroupFiniteScalarQuantizer

from pydantic import BaseModel, ValidationError

class AudioEncoderConfig(BaseModel):
    down_sample_rates: List[int] = [2, 4, 8, 8]
    encoded_dim: int = 32
    base_channels: int = 48
    activation: str = 'lrelu'

class AudioDecoderConfig(BaseModel):
    up_sample_rates: List[int] = [8, 8, 4, 2]
    input_dim: int = 32
    base_channels: int = 768
    activation: str = 'half_snake'
    output_activation: str = 'clamp'

class VectorQuantitizerConfig(BaseModel):
    num_groups: int = 8
    num_levels_per_group: List[int] = [8, 5, 5, 5]

class DiscriminatorConfig(BaseModel):
    resolutions: List[List[int]] = [[512, 128, 512], [1024, 256, 1024], [2048, 512, 2048]]
    stft_bands: List[List[float]] = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]

class AudioCodecConfig(BaseModel):
    sample_rate: int = 44100
    samples_per_frame: int = 512
    mel_loss_l1_scale: float = 10.0
    mel_loss_l2_scale: float = 0.0
    stft_loss_scale: float = 10.0
    time_domain_loss_scale: float = 0.0
    si_sdr_loss_scale: float = 0.0
    commit_loss_scale: float = 0.0
    gen_loss_scale: float = 1.0
    feature_loss_scale: float = 1.0
    disc_updates_per_period: int = 1
    disc_update_period: int = 2
    loss_resolutions: List[List[int]] = [
        [32, 8, 32],
        [64, 16, 64],
        [128, 32, 128],
        [256, 64, 256],
        [512, 128, 512],
        [1024, 256, 1024],
        [2048, 512, 2048],
    ]
    mel_loss_dims: List[int] = [5, 10, 20, 40, 80, 160, 320]
    mel_loss_log_guard: float = 1.0
    stft_loss_log_guard: float = 1.0
    feature_loss_type: Literal['absolute', 'relative'] = 'absolute'
    audio_encoder: AudioEncoderConfig = AudioEncoderConfig()
    audio_decoder: AudioDecoderConfig = AudioDecoderConfig()
    vector_quantizer: VectorQuantitizerConfig = VectorQuantitizerConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()

def load_audiocodec_config(filepath: str) -> AudioCodecConfig:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            config = AudioCodecConfig(**config_dict)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {filepath}")
    except ValidationError as e:
        raise ValidationError(f"Invalid config file format: {e}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def load_config(filepath: str) -> AudioCodecConfig:
    try:
        return load_audiocodec_config(filepath)
    except FileNotFoundError:
        print(f"Config file not found at {filepath}. Creating a default config file.")
        config = AudioCodecConfig()
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config.model_dump(), f, indent=4)
        return config


class AudioCodecModel(L.LightningModule):
    def __init__(self, cfg: AudioCodecConfig,):
        super().__init__()

        # Expected sample rate for the input audio
        self.sample_rate = cfg.sample_rate

        # Number of samples in each audio frame that is encoded
        self.samples_per_frame = cfg.samples_per_frame

        # Discriminator updates
        self.disc_updates_per_period = cfg.disc_updates_per_period
        self.disc_update_period = cfg.disc_update_period
        if self.disc_updates_per_period > self.disc_update_period:
            raise ValueError(
                f'Number of discriminator updates ({self.disc_updates_per_period}) per period must be less or equal to the configured period ({self.disc_update_period})'
            )

        # Encoder setup
        enc_cfg = cfg.audio_encoder
        self.audio_encoder = HiFiGANEncoder(encoded_dim=enc_cfg.encoded_dim, down_sample_rates=enc_cfg.down_sample_rates,
                                            base_channels=enc_cfg.base_channels, activation=enc_cfg.activation)

        # Quantizer
        vq_cfg = cfg.vector_quantizer
        self.vector_quantizer = GroupFiniteScalarQuantizer(num_groups=vq_cfg.num_groups,
                                                           num_levels_per_group=vq_cfg.num_levels_per_group)


        # Decoder setup
        dec_config = cfg.audio_decoder
        self.audio_decoder = HiFiGANDecoder(input_dim=dec_config.input_dim, up_sample_rates=dec_config.up_sample_rates,
                                            base_channels=dec_config.base_channels, activation=dec_config.activation,
                                            output_activation=dec_config.output_activation)

        # Discriminator setup
        d_cfg = cfg.discriminator
        d1 = MultiPeriodDiscriminator()
        d2 = MultiResolutionDiscriminatorSTFT(resolutions=d_cfg.resolutions, stft_bands=d_cfg.stft_bands)
        self.discriminator = Discriminator([d1, d2])

        # Mel loss setup
        loss_resolutions = cfg.loss_resolutions
        mel_loss_dims = cfg.mel_loss_dims
        mel_loss_log_guard = cfg.mel_loss_log_guard
        self.mel_loss_l1_scale = cfg.mel_loss_l1_scale
        self.mel_loss_l2_scale = cfg.mel_loss_l2_scale
        self.mel_loss_fn = MultiResolutionMelLoss(
            sample_rate=self.sample_rate,
            mel_dims=mel_loss_dims,
            resolutions=loss_resolutions,
            log_guard=mel_loss_log_guard,
        )

        # STFT loss setup
        stft_loss_log_guard = cfg.stft_loss_log_guard
        self.stft_loss_scale = cfg.stft_loss_scale
        self.stft_loss_fn = MultiResolutionSTFTLoss(
            resolutions=loss_resolutions,
            log_guard=stft_loss_log_guard,
        )

        # Time domain loss setup
        self.time_domain_loss_scale = cfg.time_domain_loss_scale
        self.si_sdr_loss_scale = cfg.si_sdr_loss_scale
        self.time_domain_loss_fn = TimeDomainLoss()
        self.si_sdr_loss_fn = SISDRLoss()

        # Discriminator loss setup
        self.gen_loss_scale = cfg.gen_loss_scale
        self.feature_loss_scale = cfg.feature_loss_scale
        self.gen_loss_fn = GeneratorSquaredLoss()
        self.disc_loss_fn = DiscriminatorSquaredLoss()

        feature_loss_type = cfg.feature_loss_type
        if feature_loss_type == "relative":
            self.feature_loss_fn = RelativeFeatureMatchingLoss()
        elif feature_loss_type == "absolute":
            self.feature_loss_fn = FeatureMatchingLoss()
        else:
            raise ValueError(f'Unknown feature loss type {feature_loss_type}.')

        # Codebook loss setup
        self.commit_loss_scale = 0.0

        # Optimizer setup
        self.lr_schedule_interval = None
        self.automatic_optimization = False

    def encode_audio(self, audio: Tensor, audio_len: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """Apply encoder on the input audio signal. Input will be padded with zeros so
        the last frame has full `self.samples_per_frame` samples.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Encoder output `encoded` and its length in number of frames `encoded_len`
        """
        if not audio_len:
            audio_len = torch.tensor([audio.shape[1]] * audio.shape[0]).to(audio.device)
        audio, audio_len = self.pad_audio(audio, audio_len)
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)
        return encoded, encoded_len

    def decode_audio(self, inputs: Tensor, input_len: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply decoder on the input. Note that the input is a non-quantized encoder output or a dequantized representation.

        Args:
            inputs: encoded signal
            input_len: valid length for each example in the batch

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        audio, audio_len = self.audio_decoder(inputs=inputs, input_len=input_len)
        return audio, audio_len

    def quantize(self, encoded: Tensor, encoded_len: Tensor) -> Tensor:
        """Quantize the continuous encoded representation into a discrete
        representation for each frame.

        Args:
            encoded: encoded signal representation
            encoded_len: valid length of the encoded representation in frames

        Returns:
            A tensor of tokens for each codebook for each frame.
        """
        # vector quantizer is returning [C, B, T], where C is the number of codebooks
        tokens = self.vector_quantizer.encode(inputs=encoded, input_len=encoded_len)
        # use batch first for the output
        tokens = rearrange(tokens, 'C B T -> B C T')
        return tokens

    def dequantize(self, tokens: Tensor, tokens_len: Tensor) -> Tensor:
        """Convert the discrete tokens into a continuous encoded representation.

        Args:
            tokens: discrete tokens for each codebook for each time frame
            tokens_len: valid length of each example in the batch

        Returns:
            Continuous encoded representation of the discrete input representation.
        """
        # vector quantizer is using [C, B, T], where C is the number of codebooks
        tokens = rearrange(tokens, 'B C T -> C B T')
        dequantized = self.vector_quantizer.decode(indices=tokens, input_len=tokens_len)
        return dequantized

    def encode(self, audio: Tensor, audio_len: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """Convert input time-domain audio signal into a discrete representation (tokens).

        Args:
            audio: input time-domain signal, shape `(batch, number of samples)`
            audio_len: valid length for each example in the batch, shape `(batch size,)`

        Returns:
            Tokens for each codebook for each frame, shape `(batch, number of codebooks, number of frames)`,
            and the corresponding valid lengths, shape `(batch,)`
        """
        # Apply encoder to obtain a continuous vector for each frame
        encoded, encoded_len = self.encode_audio(audio=audio, audio_len=audio_len)
        # Apply quantizer to obtain discrete representation per frame
        tokens = self.quantize(encoded=encoded, encoded_len=encoded_len)
        return tokens, encoded_len

    def decode(self, tokens: Tensor, tokens_len: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """Convert discrete tokens into a continuous time-domain signal.

        Args:
            tokens: discrete tokens for each codebook for each time frame, shape `(batch, number of codebooks, number of frames)`
            tokens_len: valid lengths, shape `(batch,)`

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        if not tokens_len:
            tokens_len = torch.tensor([tokens.shape[-1]] * tokens.shape[0]).to(tokens.device)
        # Convert a discrete representation to a dequantized vector for each frame
        dequantized = self.dequantize(tokens=tokens, tokens_len=tokens_len)
        # Apply decoder to obtain time-domain audio for each frame
        audio, audio_len = self.decode_audio(inputs=dequantized, input_len=tokens_len)

        return audio, audio_len

    def forward(self, audio: Tensor, audio_len: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """Apply encoder, quantizer, decoder on the input time-domain signal.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Reconstructed time-domain signal `output_audio` and its length in number of samples `output_audio_len`.
        """
        encoded, encoded_len = self.encode_audio(audio=audio, audio_len=audio_len)

        # quantize to discrete tokens
        tokens = self.quantize(encoded=encoded, encoded_len=encoded_len)
        # decode tokens to audio
        output_audio, output_audio_len = self.decode(tokens=tokens, tokens_len=encoded_len)


        return output_audio, output_audio_len

    def pad_audio(self, audio: Tensor, audio_len: Tensor) -> Tuple[Tensor, Tensor]:
        """Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `self.samples_per_frame`.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Padded time-domain signal `padded_audio` and its length `padded_len`.
        """
        padded_len = self.samples_per_frame * torch.ceil(audio_len / self.samples_per_frame).int()
        max_len = padded_len.max().item()
        num_padding = max_len - audio.shape[1]
        padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def _process_batch(self, batch):
        # [B, T_audio]
        audio = batch.get("audio")
        # [B]
        audio_len = batch.get("audio_lens")
        audio, audio_len = self.pad_audio(audio, audio_len)

        # [B, D, T_encoded]
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)

        encoded, _ = self.vector_quantizer(inputs=encoded, input_len=encoded_len)
        commit_loss = 0.0

        # [B, T]
        audio_gen, _ = self.audio_decoder(inputs=encoded, input_len=encoded_len)

        return audio, audio_len, audio_gen, commit_loss

    @property
    def disc_update_prob(self) -> float:
        """Probability of updating the discriminator."""
        return self.disc_updates_per_period / self.disc_update_period

    def should_update_disc(self, batch_idx) -> bool:
        """Decide whether to update the descriminator based
        on the batch index and configured discriminator update period.
        """
        disc_update_step = batch_idx % self.disc_update_period
        return disc_update_step < self.disc_updates_per_period

    def training_step(self, batch, batch_idx):
        optim_gen, optim_disc = self.optimizers()

        audio, audio_len, audio_gen, commit_loss = self._process_batch(batch)

        metrics = {
            "global_step": self.global_step,
            "lr": optim_gen.param_groups[0]['lr'],
        }

        if self.should_update_disc(batch_idx):
            # Train discriminator
            disc_scores_real, disc_scores_gen, _, _ = self.discriminator(audio_real=audio, audio_gen=audio_gen.detach())
            loss_disc = self.disc_loss_fn(disc_scores_real=disc_scores_real, disc_scores_gen=disc_scores_gen)
            metrics["d_loss"] = loss_disc

            optim_disc.zero_grad()
            self.manual_backward(loss_disc)
            optim_disc.step()

        generator_losses = []

        loss_mel_l1, loss_mel_l2 = self.mel_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        if self.mel_loss_l1_scale:
            metrics["g_loss_mel_l1"] = loss_mel_l1
            generator_losses.append(self.mel_loss_l1_scale * loss_mel_l1)
        if self.mel_loss_l2_scale:
            metrics["g_loss_mel_l2"] = loss_mel_l2
            generator_losses.append(self.mel_loss_l2_scale * loss_mel_l2)

        if self.stft_loss_scale:
            loss_stft = self.stft_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
            metrics["g_loss_stft"] = loss_stft
            generator_losses.append(self.stft_loss_scale * loss_stft)

        if self.time_domain_loss_scale:
            loss_time_domain = self.time_domain_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
            metrics["g_loss_time_domain"] = loss_time_domain
            generator_losses.append(self.time_domain_loss_scale * loss_time_domain)

        if self.si_sdr_loss_scale:
            loss_si_sdr = self.si_sdr_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
            metrics["g_loss_si_sdr"] = loss_si_sdr
            generator_losses.append(self.si_sdr_loss_scale * loss_si_sdr)

        _, disc_scores_gen, fmaps_real, fmaps_gen = self.discriminator(audio_real=audio, audio_gen=audio_gen)

        if self.gen_loss_scale:
            loss_gen = self.gen_loss_fn(disc_scores_gen=disc_scores_gen)
            metrics["g_loss_gen"] = loss_gen
            generator_losses.append(self.gen_loss_scale * loss_gen)

        if self.feature_loss_scale:
            loss_feature = self.feature_loss_fn(fmaps_real=fmaps_real, fmaps_gen=fmaps_gen)
            metrics["g_loss_feature"] = loss_feature
            generator_losses.append(self.feature_loss_scale * loss_feature)

        if self.commit_loss_scale:
            metrics["g_loss_commit"] = commit_loss
            generator_losses.append(self.commit_loss_scale * commit_loss)

        loss_gen_all = sum(generator_losses)

        optim_gen.zero_grad()
        self.manual_backward(loss_gen_all)
        optim_gen.step()

        self.update_lr()

        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", loss_mel_l1, prog_bar=True, logger=False, sync_dist=True)

    def on_train_epoch_end(self):
        self.update_lr("epoch")

    def validation_step(self, batch, batch_idx):
        audio, audio_len, audio_gen, _ = self._process_batch(batch)

        loss_mel_l1, loss_mel_l2 = self.mel_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        loss_stft = self.stft_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        loss_time_domain = self.time_domain_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)
        loss_si_sdr = self.si_sdr_loss_fn(audio_real=audio, audio_gen=audio_gen, audio_len=audio_len)

        # Use only main reconstruction losses for val_loss
        val_loss = loss_mel_l1 + loss_stft + loss_time_domain

        metrics = {
            "val_loss": val_loss,
            "val_loss_mel_l1": loss_mel_l1,
            "val_loss_mel_l2": loss_mel_l2,
            "val_loss_stft": loss_stft,
            "val_loss_time_domain": loss_time_domain,
            "val_loss_si_sdr": loss_si_sdr,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = 0.0002
        betas = (0.8, 0.99)
        vq_params = self.vector_quantizer.parameters() if self.vector_quantizer else []
        gen_params = itertools.chain(self.audio_encoder.parameters(), self.audio_decoder.parameters(), vq_params)
        optim_g = torch.optim.AdamW(params=gen_params, lr=lr, betas=betas)

        disc_params = self.discriminator.parameters()
        optim_d = torch.optim.AdamW(params=disc_params, lr=lr, betas=betas)

        gamma = 0.998
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_g, gamma=gamma)

        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_g, gamma=gamma)

        self.lr_schedule_interval = 'epoch'

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def update_lr(self, interval="step"):
        schedulers = self.lr_schedulers()
        if schedulers is not None and self.lr_schedule_interval == interval:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()