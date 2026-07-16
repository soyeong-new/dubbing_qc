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
    섞여 나오는 환각이 발생하고(실측 확인), condition_on_prev_tokens/no_speech_threshold
    같은 억제 옵션 없이는 비명·소음 구간에서 같은 단어를 수백 번 반복하는 환각도
    발생한다(원본 openai/whisper-large-v3-turbo로도 동일 재현 확인 — 이 모델만의 문제가
    아니라 Whisper 계열 공통 문제). temperature 폴백 시퀀스를 명시적으로 지정하지 않으면
    logprob_threshold/compression_ratio_threshold/no_speech_threshold가 내부적으로
    None과 비교 연산을 하다 크래시한다(TypeError, UnboundLocalError 둘 다 실측 확인) —
    temperature를 Whisper 표준 폴백 시퀀스로 명시하면 전부 정상 동작한다(실측 확인).
    """
    pipe = _get_pipeline()
    result = pipe(
        audio_path, return_timestamps=True,
        generate_kwargs={
            "language": "korean", "task": "transcribe",
            "condition_on_prev_tokens": False,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        },
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
