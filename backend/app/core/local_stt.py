from typing import Callable, List, Optional
from app.schemas import SegmentText

MODEL_ID = "batiai/batisay-ko-turbo"

_pipeline = None


def _select_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline
    _pipeline = pipeline(
        "automatic-speech-recognition", model=MODEL_ID,
        chunk_length_s=30, stride_length_s=5, device=_select_device(),
    )
    return _pipeline


def _run_pipeline(audio_path: str) -> list:
    """실제 로컬 Whisper 모델 호출부. transcribe_korean()의 기본 transcribe_fn이다.

    자동화 테스트는 이 함수를 절대 호출하지 않는다 — 항상 transcribe_fn을 주입해서
    실제 모델 로드를 피한다.
    """
    pipe = _get_pipeline()
    result = pipe(audio_path, return_timestamps=True)
    return result.get("chunks", [])


def transcribe_korean(
    audio_path: str,
    transcribe_fn: Optional[Callable[[str], list]] = None,
) -> List[SegmentText]:
    transcribe_fn = transcribe_fn or _run_pipeline
    chunks = transcribe_fn(audio_path)
    segments = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if not text:
            continue
        start, end = chunk["timestamp"]
        segments.append(SegmentText(
            start=float(start),
            end=float(end) if end is not None else float(start),
            text=text,
        ))
    return segments
