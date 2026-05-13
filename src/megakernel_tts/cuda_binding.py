"""Thin integration layer around the qwen_megakernel decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_megakernel.model import Decoder

from .config import QWEN3_06B_TALKER, RuntimeConfig, TalkerConfig
from .weight_adapter import load_talker_weights


@dataclass
class DecodeState:
    """Mutable state for incremental decode calls."""

    prompt_tokens: list[int]
    generated_tokens: list[int]


class MegakernelTalkerDecoder:
    """Stateful wrapper that exposes step-wise decode over the megakernel."""

    def __init__(
        self,
        model_id: str = QWEN3_06B_TALKER.model_id,
        runtime: RuntimeConfig | None = None,
        expected_config: TalkerConfig = QWEN3_06B_TALKER,
        verbose: bool = False,
    ) -> None:
        self.runtime = runtime or RuntimeConfig()
        self.expected_config = expected_config
        weights, tokenizer, actual_cfg = load_talker_weights(
            model_id=model_id,
            expected_config=expected_config,
            strict_shape_check=self.runtime.strict_shape_check,
            verbose=verbose,
        )
        self.actual_config = actual_cfg
        self.tokenizer = tokenizer
        self.decoder = Decoder(weights=weights, tokenizer=tokenizer, verbose=verbose)

    def reset(self) -> None:
        self.decoder.reset()

    def bootstrap(self, prompt: str) -> DecodeState:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(token_ids) < 1:
            raise ValueError("Prompt produced no tokens.")
        self.reset()
        for token_id in token_ids[:-1]:
            self.decoder.step(token_id)
        return DecodeState(prompt_tokens=token_ids, generated_tokens=[])

    def decode_next(self, prev_token_id: int) -> int:
        return int(self.decoder.step(prev_token_id))

    def decode_tokens(self, prompt: str, max_new_tokens: int | None = None) -> list[int]:
        budget = max_new_tokens or self.runtime.max_new_tokens
        state = self.bootstrap(prompt)
        current = state.prompt_tokens[-1]
        for _ in range(budget):
            current = self.decode_next(current)
            state.generated_tokens.append(current)
            if current == self.tokenizer.eos_token_id:
                break
        return state.generated_tokens

    def decode_text(self, prompt: str, max_new_tokens: int | None = None) -> str:
        output_ids = self.decode_tokens(prompt, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

