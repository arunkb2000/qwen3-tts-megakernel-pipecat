"""Configuration objects for megakernel-driven Qwen3-TTS integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TalkerConfig:
    """Model-shape values required by the CUDA decoder path."""

    model_id: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    torch_dtype: str = "bfloat16"

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime limits and stream behavior for the service."""

    max_new_tokens: int = 256
    audio_chunk_ms: int = 40
    sample_rate: int = 24_000
    pcm_channels: int = 1
    warmup_tokens: int = 8
    strict_shape_check: bool = True


QWEN3_06B_TALKER = TalkerConfig(
    model_id="Qwen/Qwen3-0.6B",
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151_936,
    max_position_embeddings=2048,
    rope_theta=10000.0,
)

