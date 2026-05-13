"""Load and validate Qwen talker weights for megakernel decode."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from qwen_megakernel.model import load_weights as load_megakernel_weights

from .config import TalkerConfig


def _read_hf_config(model_id: str) -> dict[str, Any]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return cfg.to_dict()


def _extract_talker_config(model_id: str) -> TalkerConfig:
    cfg = _read_hf_config(model_id)
    hidden_size = int(cfg["hidden_size"])
    num_heads = int(cfg["num_attention_heads"])
    return TalkerConfig(
        model_id=model_id,
        hidden_size=hidden_size,
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=num_heads,
        num_key_value_heads=int(cfg.get("num_key_value_heads", num_heads)),
        head_dim=hidden_size // num_heads,
        vocab_size=int(cfg["vocab_size"]),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 2048)),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
    )


def assert_compatible(actual: TalkerConfig, expected: TalkerConfig) -> None:
    """Fail fast when the target model does not match compiled kernel shapes."""
    checks = {
        "hidden_size": (actual.hidden_size, expected.hidden_size),
        "intermediate_size": (actual.intermediate_size, expected.intermediate_size),
        "num_hidden_layers": (actual.num_hidden_layers, expected.num_hidden_layers),
        "num_attention_heads": (
            actual.num_attention_heads,
            expected.num_attention_heads,
        ),
        "num_key_value_heads": (
            actual.num_key_value_heads,
            expected.num_key_value_heads,
        ),
        "head_dim": (actual.head_dim, expected.head_dim),
        "vocab_size": (actual.vocab_size, expected.vocab_size),
    }
    mismatches = {
        key: values for key, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        summary = ", ".join(
            f"{name}: actual={vals[0]} expected={vals[1]}"
            for name, vals in mismatches.items()
        )
        raise ValueError(
            "Talker shape mismatch for current megakernel build. "
            f"Model={actual.model_id}. Differences: {summary}"
        )


def load_talker_weights(
    model_id: str,
    expected_config: TalkerConfig,
    strict_shape_check: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, Any], Any, TalkerConfig]:
    """
    Return CUDA-ready weight dict + tokenizer with compatibility metadata.

    For 0.6B-compatible checkpoints this reuses the optimized loader from
    `qwen_megakernel.model` to preserve memory layout expected by the CUDA ops.
    """
    actual = _extract_talker_config(model_id)
    if strict_shape_check:
        assert_compatible(actual, expected_config)
    weights, tokenizer = load_megakernel_weights(model_name=model_id, verbose=verbose)
    weights["talker_config"] = asdict(actual)
    weights["torch_dtype"] = torch.bfloat16
    return weights, tokenizer, actual
 
