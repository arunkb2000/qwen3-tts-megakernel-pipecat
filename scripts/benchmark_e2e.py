"""Measure TTFC/RTF and end-to-end stream timings for demo pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megakernel_tts.audio_stream import AudioChunkStreamer, DummyToneBackend
from megakernel_tts.cuda_binding import MegakernelTalkerDecoder
from megakernel_tts.metrics import Timeline
from megakernel_tts.token_streamer import TokenStreamer


async def run() -> None:
    prompt = "Give me a one sentence summary of CUDA optimization."
    decoder = MegakernelTalkerDecoder(verbose=False)
    token_streamer = TokenStreamer(decoder)
    audio_streamer = AudioChunkStreamer(DummyToneBackend())
    timeline = Timeline()

    pending_ids: list[int] = []
    flush_every_tokens = 8
    sample_total = 0
    chunk_count = 0
    async for token in token_streamer.stream(prompt, max_new_tokens=120):
        timeline.mark_first_token()
        pending_ids.append(token.token_id)
        timeline.token_count += 1

        should_flush = len(pending_ids) >= flush_every_tokens or token.is_eos
        if should_flush:
            text_segment = decoder.tokenizer.decode(pending_ids, skip_special_tokens=True)
            async for chunk in audio_streamer.stream_from_token_batch(
                pending_ids,
                text_segment=text_segment,
                is_final_batch=token.is_eos,
            ):
                if timeline.first_audio_at is None:
                    timeline.mark_first_audio()
                sample_total += chunk.sample_count
                chunk_count += 1
            pending_ids = []

    if pending_ids:
        text_segment = decoder.tokenizer.decode(pending_ids, skip_special_tokens=True)
        async for chunk in audio_streamer.stream_from_token_batch(
            pending_ids,
            text_segment=text_segment,
            is_final_batch=True,
        ):
            if timeline.first_audio_at is None:
                timeline.mark_first_audio()
            sample_total += chunk.sample_count
            chunk_count += 1
    timeline.audio_seconds = sample_total / audio_streamer.sample_rate
    timeline.mark_end()

    print(f"tokens={timeline.token_count}")
    print(f"chunks={chunk_count}")
    print(f"ttfc_ms={timeline.ttfc_ms}")
    print(f"tokens_per_second={timeline.tokens_per_second:.3f}")
    print(f"rtf={timeline.rtf:.4f}")


if __name__ == "__main__":
    asyncio.run(run())

