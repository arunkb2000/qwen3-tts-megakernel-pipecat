"""Streaming token interface for prompt -> incremental decode tokens."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

from .cuda_binding import MegakernelTalkerDecoder


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    token_text: str
    step: int
    t_ms: float
    is_eos: bool


class TokenStreamer:
    """Asynchronous token stream with cancellation and backpressure support."""

    def __init__(self, decoder: MegakernelTalkerDecoder):
        self.decoder = decoder
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self._cancel.set()

    async def stream(
        self, prompt: str, max_new_tokens: int
    ) -> AsyncIterator[TokenEvent]:
        self._cancel.clear()
        started_at = time.perf_counter()
        state = self.decoder.bootstrap(prompt)
        current = state.prompt_tokens[-1]
        for step in range(max_new_tokens):
            if self._cancel.is_set():
                break
            current = self.decoder.decode_next(current)
            token_text = self.decoder.tokenizer.decode(
                [current], skip_special_tokens=False
            )
            event = TokenEvent(
                token_id=current,
                token_text=token_text,
                step=step,
                t_ms=(time.perf_counter() - started_at) * 1000.0,
                is_eos=current == self.decoder.tokenizer.eos_token_id,
            )
            yield event
            # Let other tasks run (Pipecat queue/write path).
            await asyncio.sleep(0)
            if event.is_eos:
                break

