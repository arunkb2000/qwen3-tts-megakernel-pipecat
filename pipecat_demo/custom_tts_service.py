"""Custom Pipecat TTS service backed by megakernel token decode."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megakernel_tts.audio_stream import AudioChunk, AudioChunkStreamer, DummyToneBackend
from megakernel_tts.cuda_binding import MegakernelTalkerDecoder
from megakernel_tts.qwen_tts_backend import Qwen3TTSBackend
from megakernel_tts.token_streamer import TokenStreamer

try:
    from pipecat.frames.frames import TTSAudioRawFrame
    from pipecat.services.tts_service import TTSService
except Exception:  # pragma: no cover - for environments without pipecat installed
    @dataclass
    class TTSAudioRawFrame:  # type: ignore[override]
        audio: bytes
        sample_rate: int
        num_channels: int

    class TTSService:  # type: ignore[override]
        async def push_frame(self, frame: TTSAudioRawFrame) -> None:
            _ = frame


class MegakernelQwenTTSService(TTSService):
    """Pushes PCM frames progressively to Pipecat as audio is decoded."""

    def __init__(self, max_new_tokens: int = 192, flush_every_tokens: int = 8) -> None:
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.flush_every_tokens = flush_every_tokens
        self.decoder = MegakernelTalkerDecoder(verbose=False)
        self.token_streamer = TokenStreamer(self.decoder)
        backend_name = os.getenv("MEGAKERNEL_TTS_AUDIO_BACKEND", "dummy").lower()
        strict_backend = os.getenv("MEGAKERNEL_TTS_STRICT_BACKEND", "0") == "1"
        if backend_name == "qwen3":
            try:
                audio_backend = Qwen3TTSBackend()
            except Exception as exc:
                if strict_backend:
                    raise RuntimeError(
                        "Requested qwen3 backend but qwen_tts initialization failed. "
                        "Unset MEGAKERNEL_TTS_STRICT_BACKEND or install qwen_tts."
                    ) from exc
                audio_backend = DummyToneBackend()
        else:
            audio_backend = DummyToneBackend()
        self.audio_streamer = AudioChunkStreamer(
            audio_backend,
            sample_rate=getattr(audio_backend, "sample_rate", 24_000),
        )

    async def _iter_audio(self, text: str) -> AsyncIterator[AudioChunk]:
        token_ids: list[int] = []
        async for token in self.token_streamer.stream(text, self.max_new_tokens):
            token_ids.append(token.token_id)
            should_flush = len(token_ids) >= self.flush_every_tokens or token.is_eos
            if should_flush:
                text_segment = self.decoder.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                )
                async for chunk in self.audio_streamer.stream_from_token_batch(
                    token_ids,
                    text_segment=text_segment,
                    is_final_batch=token.is_eos,
                ):
                    yield chunk
                    await asyncio.sleep(0)
                token_ids = []
        if token_ids:
            text_segment = self.decoder.tokenizer.decode(token_ids, skip_special_tokens=True)
            async for chunk in self.audio_streamer.stream_from_token_batch(
                token_ids,
                text_segment=text_segment,
                is_final_batch=True,
            ):
                yield chunk
                await asyncio.sleep(0)

    async def run_tts(self, text: str) -> None:
        started = time.perf_counter()
        emitted = 0
        async for chunk in self._iter_audio(text):
            frame = TTSAudioRawFrame(
                audio=chunk.pcm16le,
                sample_rate=chunk.sample_rate,
                num_channels=chunk.channels,
            )
            await self.push_frame(frame)
            emitted += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(f"TTS frame streaming complete: frames={emitted} elapsed_ms={elapsed_ms:.2f}")

