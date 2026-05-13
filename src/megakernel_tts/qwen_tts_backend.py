"""Optional real Qwen3-TTS backend for incremental chunk synthesis."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np


class Qwen3TTSBackend:
    """
    Wrap qwen_tts model API for text-chunk to waveform synthesis.

    This path relies on the upstream qwen_tts package. If unavailable, callers
    should fall back to DummyToneBackend.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        speaker: str = "Chelsie",
        language: str = "English",
        device_map: str = "cuda:0",
    ) -> None:
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except Exception:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # type: ignore

        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.device_map = device_map
        self.model: Any = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device_map,
        )
        # Current Qwen3-TTS outputs 24kHz in public releases.
        self.sample_rate = 24_000

    async def synthesize_chunk(self, token_ids: list[int], text_segment: str) -> np.ndarray:
        _ = token_ids
        text = text_segment.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        def _run() -> np.ndarray:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
                non_streaming_mode=False,
            )
            self.sample_rate = int(sr)
            wav = wavs[0] if isinstance(wavs, list) else wavs
            return np.asarray(wav, dtype=np.float32)

        return await asyncio.to_thread(_run)

