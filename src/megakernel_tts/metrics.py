"""Metrics helpers for decode/audio streaming benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Timeline:
    request_started_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    first_audio_at: float | None = None
    stream_ended_at: float | None = None
    token_count: int = 0
    audio_seconds: float = 0.0

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def mark_first_audio(self) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()

    def mark_end(self) -> None:
        self.stream_ended_at = time.perf_counter()

    @property
    def ttfc_ms(self) -> float | None:
        if self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.request_started_at) * 1000.0

    @property
    def tokens_per_second(self) -> float:
        if not self.stream_ended_at:
            return 0.0
        elapsed = max(1e-9, self.stream_ended_at - self.request_started_at)
        return self.token_count / elapsed

    @property
    def rtf(self) -> float:
        if not self.stream_ended_at or self.audio_seconds <= 0:
            return 0.0
        elapsed = self.stream_ended_at - self.request_started_at
        return elapsed / self.audio_seconds

