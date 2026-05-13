"""Pipecat pipeline wiring STT -> LLM -> Megakernel TTS."""

from __future__ import annotations

import asyncio
import os

from custom_tts_service import MegakernelQwenTTSService

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frameworks.llm_response import LLMResponseAggregator
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService
except Exception:
    Pipeline = None  # type: ignore[assignment]


def _has_real_pipecat_stack() -> bool:
    return Pipeline is not None


async def _run_real_pipeline() -> None:
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not deepgram_key or not openai_key:
        raise RuntimeError(
            "Set DEEPGRAM_API_KEY and OPENAI_API_KEY to run real Pipecat pipeline."
        )

    stt = DeepgramSTTService(api_key=deepgram_key)
    llm = OpenAILLMService(api_key=openai_key, model="gpt-4o-mini")
    tts = MegakernelQwenTTSService()
    agg = LLMResponseAggregator()

    pipeline = Pipeline([stt, llm, agg, tts])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    print("Starting real Pipecat pipeline (STT -> LLM -> Megakernel TTS)")
    await runner.run(task)


async def _run_stub_pipeline() -> None:
    async def fake_stt() -> str:
        return "Explain why memory bandwidth dominates decode kernels."

    async def fake_llm_reply(user_text: str) -> str:
        _ = user_text
        return (
            "Memory bandwidth dominates because each token step rereads large model "
            "weights, while arithmetic per byte is comparatively low."
        )

    tts = MegakernelQwenTTSService()
    user_text = await fake_stt()
    reply = await fake_llm_reply(user_text)
    await tts.run_tts(reply)
    print("Stub roundtrip completed: STT -> LLM -> TTS frame streaming")


async def run_demo_roundtrip() -> None:
    if _has_real_pipecat_stack():
        await _run_real_pipeline()
    else:
        await _run_stub_pipeline()


if __name__ == "__main__":
    asyncio.run(run_demo_roundtrip())

