"""Benchmark megakernel talker decode throughput and per-token latency."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megakernel_tts.cuda_binding import MegakernelTalkerDecoder

PROMPT = "Hello, this is a decode benchmark for the megakernel."
TOKENS = 128
WARMUP = 3
RUNS = 5


def run_once(decoder: MegakernelTalkerDecoder) -> float:
    started = time.perf_counter()
    _ = decoder.decode_tokens(PROMPT, max_new_tokens=TOKENS)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    decoder = MegakernelTalkerDecoder(verbose=False)
    print("Warmup...")
    for _ in range(WARMUP):
        run_once(decoder)
    print("Benchmark...")
    samples = [run_once(decoder) for _ in range(RUNS)]
    avg_s = statistics.mean(samples)
    p50_s = statistics.median(samples)
    tok_s = TOKENS / avg_s
    print(f"runs={RUNS} tokens={TOKENS}")
    print(f"avg_seconds={avg_s:.6f}")
    print(f"p50_seconds={p50_s:.6f}")
    print(f"tokens_per_second={tok_s:.2f}")
    print(f"ms_per_token={(avg_s * 1000.0 / TOKENS):.3f}")


if __name__ == "__main__":
    main()

