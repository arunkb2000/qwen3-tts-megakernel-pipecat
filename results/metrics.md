# Performance Results

Populate this file after running the benchmark scripts on RTX 5090.

Status:
- Local implementation: complete
- Hardware validation: pending RTX 5090 run
- Last updated: fill date/time after run

## Environment

- GPU: RTX 5090 (sm_120)
- CUDA: 12.8+
- Torch:
- Driver:
- Commit:

## Decode Throughput

- Command: `python scripts/benchmark_decode.py`
- Tokens/sec:
- ms/token:
- Warmup policy: 3 warmup, 5 measured runs
- Raw log: `results/logs/02_decode_bench.txt`

## Streaming TTS Metrics

- Command: `python scripts/benchmark_e2e.py`
- TTFC (request -> first audio chunk):
- RTF (generation_time / audio_duration):
- Audio chunk cadence:
- Raw log: `results/logs/03_e2e_bench.txt`

## End-to-End Pipecat Roundtrip

- Command: `python pipecat_demo/pipeline.py`
- STT -> LLM -> TTS -> output latency:
- Notes on audio quality:
- Streaming verification evidence:
  - NDJSON has monotonic `chunk_index` and increasing `chunk_t_ms`
  - Server log: `results/logs/04_server.txt`
  - Pipeline log: `results/logs/05_pipeline.txt`

## Bottlenecks and Next Improvements

- Dominant bottleneck:
- Planned optimization:

