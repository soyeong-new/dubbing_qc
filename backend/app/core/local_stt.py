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

    오디오는 무음 기준으로 직접 잘라 개별 호출하지 않고 전체 파일을 한 번에 넣는다 —
    비명/소음 구간만 통째로 떼어내 별도 호출하면 그 구간 안에 모델이 참고할 진짜
    발화가 전혀 없어 반복 환각이 오히려 심해지는 것을 실측으로 확인했다(같은 억제
    설정인데도 "아아아아..." 식 반복이 여러 구간에서 여러 번 나타남). 전체를 한 번에
    넣어 파이프라인 자체의 30초+겹침 청킹에 맡기면, 문제 구간도 앞뒤에 진짜 발화가
    있는 하나의 연속된 흐름 안에 놓여 훨씬 안정적으로 디코딩된다(같은 설정으로 반복
    환각이 사실상 사라짐을 실측 확인).
    """
    pipe = _get_pipeline()
    # return_timestamps="word": 문장(청크) 단위가 아니라 단어 단위로 타임코드를 받는다.
    # 이렇게 해야 뒤의 align 단계에서 영어 자막 한 줄 한 줄의 시간 구간에 실제로 나온
    # 한국어 단어들만 골라 붙일 수 있다 — 문장 단위 타임코드만 받으면 덩어리 전체가
    # 영어 한 줄에 통째로 붙어 검수(페르소나 패널)가 "안 맞는다"고 오판한다.
    result = pipe(
        audio_path, return_timestamps="word",
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


_SENTENCE_END = (".", "?", "!", "…")


def _group_words_into_sentences(words: list, offset: float = 0.0) -> List[SegmentText]:
    """단어 단위 타임코드 리스트를 문장 끝 부호(. ? ! …) 기준으로 문장으로 묶고,
    각 문장에 그 문장을 구성한 단어들의 절대 타임코드(첫 단어 시작 ~ 끝 단어 끝)를
    부여한다. offset은 기본 0.0 — transcribe_korean이 오디오 전체를 한 번에 넣으므로
    Whisper가 돌려주는 타임스탬프가 이미 원본 전체 타임라인 기준 절대 시간이다."""
    sentences: List[SegmentText] = []
    cur_texts: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    for w in words:
        text = w["text"].strip()
        if not text:
            continue
        rel_start, rel_end = w["timestamp"]
        rel_end = rel_end if rel_end is not None else rel_start
        abs_start = offset + float(rel_start)
        abs_end = offset + float(rel_end)
        if cur_start is None:
            cur_start = abs_start
        # Whisper가 단어 종료 타임을 시작과 같게(0초 길이) 붕괴시키는 경우가 있어
        # 뒤 단어의 end가 앞보다 작아질 수 있으므로 max로 단조 증가를 보장한다.
        cur_end = abs_end if cur_end is None else max(cur_end, abs_end)
        cur_texts.append(text)
        if text[-1] in _SENTENCE_END:
            sentences.append(SegmentText(
                start=round(cur_start, 3), end=round(cur_end, 3),
                text=" ".join(cur_texts),
            ))
            cur_texts, cur_start, cur_end = [], None, None
    if cur_texts:  # 문장 끝 부호 없이 오디오가 끝난 나머지 단어들도 한 문장으로
        sentences.append(SegmentText(
            start=round(cur_start, 3), end=round(cur_end, 3),
            text=" ".join(cur_texts),
        ))
    return sentences


def transcribe_korean(
    audio_path: str,
    transcribe_fn: Optional[Callable[[str], list]] = None,
) -> List[SegmentText]:
    """원본 오디오 전체를 한 번에 Whisper에 넣어(무음 기준으로 직접 잘라 개별
    호출하지 않음 — _run_pipeline의 docstring 참고, 비명/소음 구간을 통째로 떼어
    별도 호출하면 반복 환각이 오히려 심해짐을 실측 확인) 단어 단위로 전사한다.
    검수 단위로는 단어가 너무 잘게 쪼개져 부자연스러우므로 문장 끝 부호 기준으로
    다시 문장으로 묶는다. Whisper가 돌려주는 타임스탬프가 이미 원본 전체 타임라인
    기준이므로 별도 오프셋 복원이 필요 없다.

    결과적으로 문장 단위 + 정확한 타임코드가 되어, 뒤의 align 단계가 영어 자막 줄의
    시간 구간에 맞는 한국어 문장을 정확히 골라 붙일 수 있다.
    """
    transcribe_fn = transcribe_fn or _run_pipeline
    words = transcribe_fn(audio_path)
    return _group_words_into_sentences(words)
