"""Convert token stream events into frame-by-frame PCM chunks."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import numpy as np


class IncrementalAudioBackend(Protocol):
    """Backend that converts token/text chunks into mono float32 PCM."""

    sample_rate: int

    async def synthesize_chunk(
        self, token_ids: list[int], text_segment: str
    ) -> np.ndarray:
        """Return mono float32 PCM in range [-1, 1]."""

class DummyToneBackend:
    """
    Smoke-test backend used until Qwen3-TTS backend is wired.

    Produces a deterministic tone sequence keyed by token values, allowing
    end-to-end streaming tests without a full neural vocoder dependency.
    """

    def __init__(self, sample_rate: int = 24_000):
        self.sample_rate = sample_rate

    async def synthesize_chunk(self, token_ids: list[int], text_segment: str) -> np.ndarray:
        _ = text_segment
        tone_ms = 10
        chunk = int(self.sample_rate * tone_ms / 1000)
        out = np.zeros(chunk * max(len(token_ids), 1), dtype=np.float32)
        for idx, token in enumerate(token_ids):
            freq = 240.0 + (token % 400)
            t = np.arange(chunk, dtype=np.float32) / self.sample_rate
            out[idx * chunk : (idx + 1) * chunk] = 0.1 * np.sin(2.0 * math.pi * freq * t)
        await asyncio.sleep(0)
        return out


@dataclass(frozen=True)
class AudioChunk:
    pcm16le: bytes
    sample_rate: int
    channels: int
    sample_count: int
    is_final: bool


class AudioChunkStreamer:
    """Incrementally emit PCM chunks for Pipecat frame-by-frame push."""

    def __init__(
        self,
        backend: IncrementalAudioBackend,
        sample_rate: int = 24_000,
        channels: int = 1,
        chunk_ms: int = 40,
    ) -> None:
        self.backend = backend
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

    async def stream_from_token_batch(
        self, token_ids: list[int], text_segment: str, is_final_batch: bool
    ) -> AsyncIterator[AudioChunk]:
        pcm = await self.backend.synthesize_chunk(token_ids=token_ids, text_segment=text_segment)
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        if len(pcm16) == 0:
            return
        for start in range(0, len(pcm16), self.chunk_samples):
            end = min(start + self.chunk_samples, len(pcm16))
            yield AudioChunk(
                pcm16le=pcm16[start:end].tobytes(),
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_count=end - start,
                is_final=is_final_batch and end >= len(pcm16),
            )
            await asyncio.sleep(0)

