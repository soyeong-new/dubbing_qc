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

    모델 카드(batiai/batisay-ko-turbo)가 명시적으로 경고하는 대로, 장편(60분+) 오디오는
    언어를 고정하지 않고 이전 문맥에 조건화한 채로 디코딩하면 무음/전환 구간에서
    반복 환각·다른 언어 혼입("디코더 붕괴")이 발생한다. language를 한국어로 고정하고
    condition_on_prev_tokens를 꺼서 각 청크를 독립적으로 디코딩하고, no_speech_threshold로
    무음 구간을 억제한다.

    주의: logprob_threshold/compression_ratio_threshold는 추가하지 않는다 — 이 두 옵션은
    transformers의 온도(temperature) 폴백 디코딩 로직을 함께 활성화해야 하는데, temperature를
    별도로 지정하지 않으면 내부적으로 None과 float를 비교하다 TypeError로 크래시한다
    (실측 확인됨). 모델 카드도 이 두 옵션을 요구하지 않는다.
    """
    pipe = _get_pipeline()
    result = pipe(
        audio_path, return_timestamps=True,
        generate_kwargs={
            "language": "korean", "task": "transcribe",
            "condition_on_prev_tokens": False,
            "no_speech_threshold": 0.6,
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
