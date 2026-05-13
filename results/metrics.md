# Performance Results

Measured on rented RTX 5090; logs committed under `results/logs/`.

Status:

- Local implementation: complete
- Hardware validation: completed (2026-05-13 UTC)
- Last updated: 2026-05-13T19:32:23Z (capture time on GPU)

## Environment

- GPU: NVIDIA GeForce RTX 5090 (sm_120), 32607 MiB
- CUDA (driver report): 12.8
- Driver: 570.172.08
- Torch: 2.8.0+cu128, CUDA runtime 12.8, `torch.cuda.is_available()`: True
- Commit: `1c13e8422260ff4512592a7ff3897eafad3c62b2`
- Host note: Vast.ai-style instance (`root@C.36700177`)

## Decode Throughput

- Command: `python scripts/benchmark_decode.py` (from repo root, `.venv` active)
- Workload: `PROMPT` + `max_new_tokens=128` per timed run (see script)
- Tokens/sec: **933.52** (aggregate over 128 generated tokens per run)
- ms/token: **1.071**
- Run aggregate timing: avg **0.137116** s per 128-token decode (5 runs after warmup)
- Warmup policy: **3** warmup runs, **5** measured runs (`scripts/benchmark_decode.py`)
- Raw log: `results/logs/02_decode_bench.txt`

## Streaming TTS Metrics

- Command: `python scripts/benchmark_e2e.py`
- Harness settings (from run): **120** talker tokens, **30** audio chunk events
- TTFC (request → first audio chunk): **~18.6 ms**
- Decode throughput (harness-reported): **~922 tok/s**
- RTF: **0.1084** (generation wall time / produced audio duration, per harness)
- Raw log: `results/logs/03_e2e_bench.txt`

## End-to-End Pipecat Roundtrip

- Command: `MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3 MEGAKERNEL_TTS_STRICT_BACKEND=1 python pipecat_demo/pipeline.py`
- Configuration: **stub** STT and LLM path in `pipecat_demo/pipeline.py` (no Deepgram/OpenAI keys on this run); real **Qwen3-TTS** backend for audio generation
- Observed run: **2006** `TTSAudioRawFrame` emissions, wall clock **~56028 ms** for the full synthetic utterance (see log tail)
- Notes on audio quality: listen-back on this run was not part of automated capture; `qwen3` path completed without speaker/model errors after switching to supported model/speaker. Pipecat warned **flash-attn** not installed (PyTorch fallback only).
- Streaming verification evidence:
  - Server path: NDJSON `audio_chunk` events include monotonic `chunk_index` and `chunk_t_ms` (see `src/megakernel_tts/server.py`); raw capture: `results/logs/04_server.txt`
  - Pipeline log: `results/logs/05_pipeline.txt`

## Bottlenecks and Next Improvements

- Dominant bottleneck: **End-to-end TTS and frame emission** dominate wall time versus raw megakernel decode in isolation; full `Qwen3TTS` synthesis per micro-batch is heavier than token generation alone.
- Planned optimization: optional **flash-attn** install for `qwen_tts` stack; tighter coupling of talker semantic tokens to codec streaming (per README “Known Gaps”) to reduce per-batch synthesis overhead; real STT/LLM keys for a true spoken round-trip demo.

## Other logs

- Preflight: `results/logs/00_preflight.txt`
- Upstream megakernel package bench: `results/logs/01_megakernel_bench.txt`
