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

    언어를 자동 감지에 맡기면 무음/전환 구간에서 다른 언어(중국어·일본어·프랑스어 등)가
    섞여 나오는 환각이 발생한다(실측 확인) — language를 한국어로 고정해 이를 막는다.

    주의: condition_on_prev_tokens=False나 no_speech_threshold 같은 폴백/억제 옵션은
    일부러 넣지 않는다 — 이 특정 transformers 버전에서 모델에 내장된 generation_config와
    상호작용하며 내부적으로 크래시(TypeError, UnboundLocalError)를 일으키는 것을 실측으로
    확인했다. 언어 고정만으로 관찰된 핵심 문제(다른 언어 혼입)는 해결되며, 나머지
    환각 억제는 이 버전 조합에서는 불안정해 보류한다.
    """
    pipe = _get_pipeline()
    result = pipe(
        audio_path, return_timestamps=True,
        generate_kwargs={"language": "korean", "task": "transcribe"},
    )
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
