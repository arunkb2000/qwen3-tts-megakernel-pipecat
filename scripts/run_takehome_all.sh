#!/usr/bin/env bash
set -euo pipefail

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

mkdir -p results/logs

echo "[run] preflight"
python3 scripts/validate_takehome.py | tee results/logs/00_preflight.txt

echo "[run] baseline megakernel benchmark"
python3 -m qwen_megakernel.bench | tee results/logs/01_megakernel_bench.txt

echo "[run] integration decode benchmark"
python3 scripts/benchmark_decode.py | tee results/logs/02_decode_bench.txt

echo "[run] e2e benchmark harness"
python3 scripts/benchmark_e2e.py | tee results/logs/03_e2e_bench.txt

echo
echo "[run] Next steps (separate terminals):"
echo "  1) MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3 uvicorn megakernel_tts.server:app --app-dir src --host 0.0.0.0 --port 8000 | tee results/logs/04_server.txt"
echo "  2) python3 pipecat_demo/pipeline.py | tee results/logs/05_pipeline.txt"
echo
echo "[run] Fill results/metrics.md using logs in results/logs/"
