"""Run a practical validation checklist and print next actions.

This script is intentionally lightweight so it can run before heavyweight model
downloads. It verifies environment assumptions and prints exact commands for
the full benchmark/report flow.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as exc:
        return exc.returncode, exc.output.strip()
    except FileNotFoundError:
        return 127, ""


def main() -> None:
    print("== Take-home validation preflight ==")
    print(f"Platform: {platform.platform()}")
    print(f"Workspace: {ROOT}")
    print()

    required_bins = ["python3", "nvidia-smi"]
    for name in required_bins:
        print(f"{name}: {'ok' if shutil.which(name) else 'missing'}")

    code, out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if code == 0:
        print(f"Detected GPU(s): {out}")
    else:
        print("Detected GPU(s): unavailable from nvidia-smi")

    print()
    deps = [
        "torch",
        "transformers",
        "fastapi",
        "uvicorn",
        "numpy",
        "pipecat",
        "qwen_tts",
    ]
    for dep in deps:
        print(f"python module {dep}: {'ok' if _has_module(dep) else 'missing'}")

    print()
    print("== Full execution commands (run on RTX 5090 host) ==")
    print("bash scripts/setup_5090.sh")
    print("python3 -m pip install -r requirements.txt")
    print("python3 -m qwen_megakernel.bench")
    print("python3 scripts/benchmark_decode.py")
    print("python3 scripts/benchmark_e2e.py")
    print(
        "MEGAKERNEL_TTS_AUDIO_BACKEND=qwen3 uvicorn megakernel_tts.server:app "
        "--app-dir src --host 0.0.0.0 --port 8000"
    )
    print("python3 pipecat_demo/pipeline.py")
    print("bash scripts/run_takehome_all.sh")

    print()
    print("If qwen_tts is missing, install according to the upstream Qwen3-TTS repo.")
    print("The setup script includes both pip and GitHub fallback install attempts.")
    print("Then re-run benchmark scripts and update results/metrics.md with observed values.")


if __name__ == "__main__":
    main()

