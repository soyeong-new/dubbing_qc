import math
from typing import Callable, List, Optional, Tuple
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


def _rms(chunk) -> float:
    if not chunk:
        return 0.0
    return math.sqrt(sum(s * s for s in chunk) / len(chunk))


def _detect_speech_windows(
    samples, rate: int, silence_threshold: float = 100, min_silence_s: float = 0.5,
    max_window_s: float = 25.0, frame_ms: int = 100,
) -> List[Tuple[float, float]]:
    """RMS 에너지로 무음 구간을 찾아, 그 사이의 발화 구간을 (start, end) 절대 초 단위로
    반환한다. Whisper의 30초 입력 제약을 피하기 위해 한 발화 구간이 max_window_s를
    넘으면 강제로 나눈다 — 짧은 자연스러운 휴지(min_silence_s 미만)는 발화가 이어지는
    것으로 보고 하나로 유지한다(문장 중간의 숨쉬기 등으로 과도하게 쪼개지지 않도록).
    """
    frame = max(1, int(rate * frame_ms / 1000))
    frame_count = len(samples) // frame or 1
    min_silence_frames = max(1, int(min_silence_s * 1000 / frame_ms))

    raw_windows: List[Tuple[int, int]] = []
    seg_start = None
    silent_run = 0
    for i in range(frame_count):
        chunk = samples[i * frame:(i + 1) * frame]
        if _rms(chunk) >= silence_threshold:
            if seg_start is None:
                seg_start = i
            silent_run = 0
        elif seg_start is not None:
            silent_run += 1
            if silent_run >= min_silence_frames:
                raw_windows.append((seg_start, i - silent_run + 1))
                seg_start = None
                silent_run = 0
    if seg_start is not None:
        raw_windows.append((seg_start, frame_count))

    windows: List[Tuple[float, float]] = []
    for s, e in raw_windows:
        start_s = s * frame_ms / 1000.0
        end_s = e * frame_ms / 1000.0
        while end_s - start_s > max_window_s:
            windows.append((start_s, start_s + max_window_s))
            start_s += max_window_s
        if end_s > start_s:
            windows.append((start_s, end_s))
    return windows


def transcribe_korean(
    audio_path: str,
    transcribe_fn: Optional[Callable[[str], list]] = None,
    extract_clip_fn: Optional[Callable[[str, float, float], str]] = None,
) -> List[SegmentText]:
    """원본 오디오를 무음 기준으로 발화 구간마다 잘라(각 구간은 Whisper의 30초 제약
    안에 들도록 보장됨) 개별적으로 전사한 뒤, 각 구간의 원래 절대 시작 시각을
    Whisper가 돌려준 클립 상대 타임스탬프에 다시 더해 원본 전체 타임라인 기준
    타임코드로 복원한다. 이 오프셋 복원을 빠뜨리면 모든 구간이 "0초부터 시작한
    것처럼" 잘못된 타임코드를 갖게 되므로 반드시 필요하다.
    """
    from app.core.rule_checks import read_wav_mono, extract_clip as default_extract_clip

    transcribe_fn = transcribe_fn or _run_pipeline
    extract_clip_fn = extract_clip_fn or default_extract_clip

    samples, rate = read_wav_mono(audio_path)
    windows = _detect_speech_windows(samples, rate)

    segments: List[SegmentText] = []
    for window_start, window_end in windows:
        clip_path = extract_clip_fn(audio_path, window_start, window_end)
        chunks = transcribe_fn(clip_path)
        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                continue
            rel_start, rel_end = chunk["timestamp"]
            rel_end = rel_end if rel_end is not None else rel_start
            segments.append(SegmentText(
                start=round(window_start + float(rel_start), 3),
                end=round(window_start + float(rel_end), 3),
                text=text,
            ))
    return segments
