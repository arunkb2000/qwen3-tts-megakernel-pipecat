## Megakernel -> Qwen3-TTS -> Pipecat

This repository integrates AlpinDale's RTX 5090 Qwen megakernel decode path into a
streaming TTS stack intended for Pipecat voice pipelines.

The focus is to accelerate the **talker decoder stage** using the CUDA megakernel and
stream audio chunks continuously (no full-utterance buffering).

Reference:
- [AlpinDale decode optimization blog](https://blog.alpindale.net/posts/5090_decode_optimization/)
- [qwen_megakernel source](https://github.com/AlpinDale/qwen_megakernel)
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS)
- [Pipecat docs](https://docs.pipecat.ai)

## Architecture

1. Prompt text enters the megakernel-backed talker decoder.
2. Tokens are streamed incrementally through `TokenStreamer`.
3. Tokens are converted to PCM chunks through `AudioChunkStreamer`.
4. Pipecat custom service emits `TTSAudioRawFrame` chunks frame-by-frame.

Current implementation supports two audio backends:

- `dummy` (default): deterministic tone backend for smoke tests and profiling.
- `qwen3`: real Qwen3-TTS generation path (requires `qwen_tts` runtime).

Set `MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3` to enable the real backend.
Set `MEGAKERNEL_TTS_STRICT_BACKEND=1` to fail fast instead of falling back to dummy.

## Project Layout

- `qwen_megakernel/`: original decode kernel and Python wrapper
- `src/megakernel_tts/config.py`: model/runtime config and compatibility defaults
- `src/megakernel_tts/weight_adapter.py`: talker config extraction + strict shape checks
- `src/megakernel_tts/cuda_binding.py`: megakernel talker wrapper
- `src/megakernel_tts/token_streamer.py`: prompt -> token stream
- `src/megakernel_tts/audio_stream.py`: token stream -> chunked PCM
- `src/megakernel_tts/server.py`: NDJSON streaming inference endpoint
- `pipecat_demo/custom_tts_service.py`: Pipecat TTS service emitting audio frames
- `pipecat_demo/pipeline.py`: end-to-end STT -> LLM -> TTS demo skeleton
- `scripts/benchmark_decode.py`: talker decode tok/s benchmark
- `scripts/benchmark_e2e.py`: TTFC/RTF benchmark harness
- `results/metrics.md`: performance report template

## Environment

- GPU: RTX 5090 (`sm_120a`)
- CUDA: 12.8+
- Python: 3.10+ recommended

Install:

```bash
python -m pip install -r requirements.txt
```

Optional helper on cloud RTX 5090:

```bash
bash scripts/setup_5090.sh
```

## Run

### 1) Baseline megakernel benchmark

```bash
python -m qwen_megakernel.bench
```

### 2) Decode benchmark for integration path

```bash
python scripts/benchmark_decode.py
```

### 3) End-to-end stream metrics (TTFC/RTF harness)

```bash
python scripts/benchmark_e2e.py
```

### 3.5) Assignment preflight checklist

```bash
python3 scripts/validate_takehome.py
```

### 4) Streaming inference server

```bash
uvicorn megakernel_tts.server:app --app-dir src --host 0.0.0.0 --port 8000
```

Enable real Qwen3-TTS backend:

```bash
MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3 uvicorn megakernel_tts.server:app --app-dir src --host 0.0.0.0 --port 8000
```

The stream endpoint emits `audio_chunk` events with `chunk_index` and `chunk_t_ms`,
so you can prove frame-by-frame streaming behavior in logs.

### 5) Pipecat demo pipeline

```bash
python pipecat_demo/pipeline.py
```

For a real Pipecat provider chain, set:

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3 MEGAKERNEL_TTS_STRICT_BACKEND=1 python pipecat_demo/pipeline.py
```

### 6) One-command benchmark pass

```bash
bash scripts/run_takehome_all.sh
```

## Kernel Modification Notes

This branch keeps kernel-level CUDA logic unchanged from the megakernel baseline and
focuses on integration interfaces around it:

- Strict shape guard for talker checkpoint compatibility (`weight_adapter.py`)
- Streaming token interface for incremental decode (`token_streamer.py`)
- Frame-by-frame PCM chunk emission for Pipecat with micro-batch flushing (`audio_stream.py`, `custom_tts_service.py`)

If you target a non-0.6B talker variant, adapt kernel constants in `csrc/kernel.cu`
and update shape checks in `src/megakernel_tts/config.py`.

## Performance Reporting

Run the benchmark scripts and fill `results/metrics.md` with:

- decode tokens/s
- TTFC (target < 60 ms)
- RTF (target < 0.15)
- end-to-end roundtrip latency
- streaming evidence (chunk cadence)

Suggested evidence capture:
- Save server NDJSON output and show monotonic `chunk_index`.
- Plot or tabulate inter-chunk `chunk_t_ms` deltas to prove incremental output.
- Keep raw logs under `results/logs/` and reference them in your submission.

## Cloud RTX 5090 Quickstart

Use this when you do not have local 5090 hardware:

1. Rent an RTX 5090 host (CUDA 12.8+).
2. Clone this repo and run:
   - `bash scripts/setup_5090.sh`
   - `bash scripts/run_takehome_all.sh`
3. Start the streaming server:
   - `MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3 uvicorn megakernel_tts.server:app --app-dir src --host 0.0.0.0 --port 8000`
4. Run the demo pipeline:
   - `python3 pipecat_demo/pipeline.py`
5. Copy results from `results/logs/` into `results/metrics.md`.

## Known Gaps / Next Steps

- Replace text-chunk fallback synthesis with true codec-level incremental decode from the talker output path.
- Wire production transport/input/output in Pipecat (the current file provides provider services but not app-specific transport integration).

