#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Starting RTX 5090 environment setup"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] python3 is required"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] nvidia-smi not found; ensure this host has NVIDIA drivers"
  exit 1
fi

echo "[setup] GPU info:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ ! -d ".venv" ]]; then
  echo "[setup] creating virtualenv .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -r requirements.txt

echo "[setup] installing qwen_tts (primary attempt)"
if ! python3 -m pip install qwen-tts; then
  echo "[setup] qwen-tts package not available; trying upstream GitHub repo"
  if ! python3 -m pip install "git+https://github.com/QwenLM/Qwen3-TTS.git"; then
    echo "[setup] could not auto-install qwen_tts. Please follow upstream Qwen3-TTS install docs."
  fi
fi

echo "[setup] preflight checks"
python3 scripts/validate_takehome.py

echo "[setup] complete"
echo "[setup] activate environment with: source .venv/bin/activate"
