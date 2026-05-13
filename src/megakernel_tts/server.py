"""Streaming inference server: prompt in -> token/audio streams out."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .audio_stream import AudioChunkStreamer, DummyToneBackend
from .cuda_binding import MegakernelTalkerDecoder
from .metrics import Timeline
from .qwen_tts_backend import Qwen3TTSBackend
from .token_streamer import TokenStreamer


class SynthesizeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    flush_every_tokens: int = Field(default=8, ge=1, le=64)


def create_app() -> FastAPI:
    app = FastAPI(title="Megakernel Qwen3-TTS Stream Server")
    decoder = MegakernelTalkerDecoder(verbose=False)
    token_streamer = TokenStreamer(decoder)
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
    audio_streamer = AudioChunkStreamer(
        backend=audio_backend,
        sample_rate=getattr(audio_backend, "sample_rate", 24_000),
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/stream")
    async def stream(req: SynthesizeRequest) -> StreamingResponse:
        timeline = Timeline()

        async def generate() -> AsyncIterator[bytes]:
            pending_ids: list[int] = []
            sample_total = 0
            chunk_index = 0
            request_t0 = time.perf_counter()
            async for token in token_streamer.stream(req.prompt, req.max_new_tokens):
                timeline.mark_first_token()
                timeline.token_count += 1
                pending_ids.append(token.token_id)
                yield (
                    json.dumps(
                        {
                            "type": "token",
                            "step": token.step,
                            "token_id": token.token_id,
                            "token_text": token.token_text,
                            "t_ms": round(token.t_ms, 3),
                            "is_eos": token.is_eos,
                        }
                    )
                    + "\n"
                ).encode("utf-8")

                should_flush = (
                    len(pending_ids) >= req.flush_every_tokens or token.is_eos
                )
                if should_flush:
                    text_segment = decoder.tokenizer.decode(
                        pending_ids, skip_special_tokens=True
                    )
                    async for chunk in audio_streamer.stream_from_token_batch(
                        pending_ids,
                        text_segment=text_segment,
                        is_final_batch=token.is_eos,
                    ):
                        if timeline.first_audio_at is None:
                            timeline.mark_first_audio()
                        sample_total += chunk.sample_count
                        yield (
                            json.dumps(
                                {
                                    "type": "audio_chunk",
                                    "sample_rate": chunk.sample_rate,
                                    "channels": chunk.channels,
                                    "sample_count": chunk.sample_count,
                                    "is_final": chunk.is_final,
                                    "chunk_index": chunk_index,
                                    "chunk_t_ms": round(
                                        (time.perf_counter() - request_t0) * 1000.0, 3
                                    ),
                                    "pcm16le_b64": base64.b64encode(chunk.pcm16le).decode(
                                        "ascii"
                                    ),
                                }
                            )
                            + "\n"
                        ).encode("utf-8")
                        chunk_index += 1
                    pending_ids = []

            if pending_ids:
                text_segment = decoder.tokenizer.decode(
                    pending_ids, skip_special_tokens=True
                )
                async for chunk in audio_streamer.stream_from_token_batch(
                    pending_ids,
                    text_segment=text_segment,
                    is_final_batch=True,
                ):
                    if timeline.first_audio_at is None:
                        timeline.mark_first_audio()
                    sample_total += chunk.sample_count
                    yield (
                        json.dumps(
                            {
                                "type": "audio_chunk",
                                "sample_rate": chunk.sample_rate,
                                "channels": chunk.channels,
                                "sample_count": chunk.sample_count,
                                "is_final": chunk.is_final,
                                "chunk_index": chunk_index,
                                "chunk_t_ms": round(
                                    (time.perf_counter() - request_t0) * 1000.0, 3
                                ),
                                "pcm16le_b64": base64.b64encode(chunk.pcm16le).decode(
                                    "ascii"
                                ),
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                    chunk_index += 1

            timeline.audio_seconds = sample_total / float(audio_streamer.sample_rate)
            timeline.mark_end()
            yield (
                json.dumps(
                    {
                        "type": "metrics",
                        "ttfc_ms": timeline.ttfc_ms,
                        "tokens_per_second": timeline.tokens_per_second,
                        "rtf": timeline.rtf,
                    }
                )
                + "\n"
            ).encode("utf-8")

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    return app


app = create_app()

