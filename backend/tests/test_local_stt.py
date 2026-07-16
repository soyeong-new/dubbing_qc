import math
import struct
import wave

import app.core.local_stt as local_stt
from app.core.local_stt import transcribe_korean, _detect_speech_windows


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"{len(samples)}h", *samples))


def sine(seconds, rate=16000, amp=8000, freq=440):
    n = int(seconds * rate)
    return [int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]


def silence(seconds, rate=16000):
    return [0] * int(seconds * rate)


# ---- _detect_speech_windows: 무음 기반 발화 구간 탐지 ----

def test_detect_speech_windows_splits_on_silence_gap():
    # 0~2초 무음, 2~5초 발화, 5~10초 무음, 10~14초 발화, 14~16초 무음
    samples = silence(2) + sine(3) + silence(5) + sine(4) + silence(2)
    windows = _detect_speech_windows(samples, rate=16000, silence_threshold=100, min_silence_s=0.5)
    assert len(windows) == 2
    assert windows[0] == (2.0, 5.0)
    assert windows[1] == (10.0, 14.0)


def test_detect_speech_windows_tolerates_short_natural_pause():
    # 발화 중간에 0.2초의 짧은 휴지(min_silence_s=0.5보다 짧음)가 있어도
    # 하나의 이어지는 발화 구간으로 유지되어야 한다(문장 중간 숨쉬기 등).
    samples = silence(1) + sine(2) + silence(0.2) + sine(2) + silence(1)
    windows = _detect_speech_windows(samples, rate=16000, silence_threshold=100, min_silence_s=0.5)
    assert len(windows) == 1
    assert windows[0][0] == 1.0
    assert windows[0][1] == 5.2  # 1(무음) + 2(발화) + 0.2(짧은 휴지) + 2(발화)


def test_detect_speech_windows_force_splits_long_uninterrupted_speech():
    # 무음 없이 30초 넘게 이어지는 발화는 Whisper의 30초 제약을 피하기 위해
    # max_window_s 단위로 강제 분할되어야 한다.
    samples = sine(40)
    windows = _detect_speech_windows(samples, rate=16000, max_window_s=25.0)
    assert len(windows) == 2
    assert windows[0] == (0.0, 25.0)
    assert windows[1] == (25.0, 40.0)


def test_detect_speech_windows_empty_audio_returns_no_windows():
    samples = silence(3)
    windows = _detect_speech_windows(samples, rate=16000)
    assert windows == []


# ---- _group_words_into_sentences: 단어 → 문장 묶기 ----

def test_group_words_into_sentences_splits_on_sentence_end():
    from app.core.local_stt import _group_words_into_sentences
    words = [
        {"text": " 밥", "timestamp": (0.0, 0.4)},
        {"text": " 먹었어?", "timestamp": (0.4, 1.0)},
        {"text": " 어이가", "timestamp": (1.5, 2.0)},
        {"text": " 없네.", "timestamp": (2.0, 2.6)},
    ]
    sents = _group_words_into_sentences(words, offset=0.0)
    assert len(sents) == 2
    assert sents[0].text == "밥 먹었어?"
    assert sents[0].start == 0.0 and sents[0].end == 1.0
    assert sents[1].text == "어이가 없네."
    assert sents[1].start == 1.5 and sents[1].end == 2.6


def test_group_words_into_sentences_applies_offset():
    from app.core.local_stt import _group_words_into_sentences
    words = [{"text": " 대사.", "timestamp": (0.2, 0.9)}]
    sents = _group_words_into_sentences(words, offset=100.0)
    assert sents[0].start == 100.2 and sents[0].end == 100.9


def test_group_words_into_sentences_flushes_trailing_words_without_punctuation():
    from app.core.local_stt import _group_words_into_sentences
    words = [
        {"text": " 끝나지", "timestamp": (0.0, 0.5)},
        {"text": " 않은", "timestamp": (0.5, 1.0)},
    ]
    sents = _group_words_into_sentences(words, offset=0.0)
    assert len(sents) == 1
    assert sents[0].text == "끝나지 않은"


def test_group_words_into_sentences_guards_collapsed_word_timestamps():
    from app.core.local_stt import _group_words_into_sentences
    # Whisper가 단어 종료 타임을 시작과 같게 붕괴시켜 end가 앞 단어보다 작아질 수 있다.
    words = [
        {"text": " 첫", "timestamp": (0.0, 1.0)},
        {"text": " 단어.", "timestamp": (0.5, 0.5)},
    ]
    sents = _group_words_into_sentences(words, offset=0.0)
    assert len(sents) == 1
    assert sents[0].start == 0.0
    assert sents[0].end == 1.0  # max로 단조 증가 보장 (0.5로 뒷걸음치지 않음)


# ---- transcribe_korean: 절대 타임코드 복원 + 문장 묶기 ----

def test_transcribe_korean_groups_words_into_sentences_with_absolute_timestamps(tmp_path):
    # 무음 기준으로 잘라낸 각 구간은 별도의 클립(그 자체로는 0초부터 시작하는 파일)이
    # 되므로 Whisper가 돌려주는 단어 타임스탬프는 "클립 안에서" 상대적인 시간이다.
    # transcribe_korean은 이걸 원본 절대 시간으로 복원하고, 단어를 문장으로 묶는다.
    wav_path = tmp_path / "original.wav"
    # 0~3초 무음, 3~6초 발화(구간 A), 6~11초 무음, 11~14초 발화(구간 B)
    write_wav(wav_path, silence(3) + sine(3) + silence(5) + sine(3))

    calls = []

    def fake_extract_clip(src, start, end):
        calls.append((start, end))
        return f"clip_{start}_{end}.wav"

    def fake_transcribe(clip_path):
        # 클립 내부 기준 상대 시간의 단어 두 개가 한 문장을 이룬다.
        return [
            {"text": " 안녕", "timestamp": (0.1, 0.5)},
            {"text": " 하세요.", "timestamp": (0.5, 1.0)},
        ]

    segments = transcribe_korean(
        str(wav_path), transcribe_fn=fake_transcribe, extract_clip_fn=fake_extract_clip,
    )

    assert len(calls) == 2
    assert calls[0] == (3.0, 6.0)
    assert calls[1] == (11.0, 14.0)

    assert len(segments) == 2
    # 구간 A(절대 시작 3.0초): 단어들이 상대 0.1~1.0초 → 문장 절대 3.1~4.0초.
    assert segments[0].text == "안녕 하세요."
    assert segments[0].start == 3.1 and segments[0].end == 4.0
    # 구간 B(절대 시작 11.0초): 문장 절대 11.1~12.0초.
    assert segments[1].text == "안녕 하세요."
    assert segments[1].start == 11.1 and segments[1].end == 12.0


def test_transcribe_korean_skips_empty_words(tmp_path):
    wav_path = tmp_path / "x.wav"
    write_wav(wav_path, silence(1) + sine(2) + silence(1))

    def fake_transcribe(clip_path):
        return [
            {"text": "   ", "timestamp": (0.0, 0.3)},
            {"text": " 실제", "timestamp": (0.3, 0.8)},
            {"text": " 대사.", "timestamp": (0.8, 1.5)},
        ]

    segments = transcribe_korean(
        str(wav_path), transcribe_fn=fake_transcribe,
        extract_clip_fn=lambda src, s, e: src,
    )
    assert len(segments) == 1
    assert segments[0].text == "실제 대사."


def test_transcribe_korean_handles_open_ended_last_word(tmp_path):
    # Whisper의 마지막 단어는 종료 타임스탬프가 None일 수 있다
    wav_path = tmp_path / "x.wav"
    write_wav(wav_path, silence(1) + sine(2) + silence(1))

    def fake_transcribe(clip_path):
        return [{"text": " 마지막.", "timestamp": (0.5, None)}]

    segments = transcribe_korean(
        str(wav_path), transcribe_fn=fake_transcribe,
        extract_clip_fn=lambda src, s, e: src,
    )
    assert len(segments) == 1
    assert segments[0].start == 1.5  # window_start(1.0) + rel_start(0.5)
    assert segments[0].end == 1.5  # end가 없으면 start로 대체


def test_transcribe_korean_does_not_import_transformers_at_module_load(monkeypatch, tmp_path):
    # transcribe_fn을 주입하면 실제 모델 로드 경로(_get_pipeline)가 전혀 호출되지 않아야 한다
    wav_path = tmp_path / "x.wav"
    write_wav(wav_path, silence(1) + sine(2) + silence(1))

    def boom():
        raise AssertionError("실제 모델을 로드하면 안 된다")

    monkeypatch.setattr(local_stt, "_get_pipeline", boom)
    segments = transcribe_korean(
        str(wav_path),
        transcribe_fn=lambda p: [{"text": " ok.", "timestamp": (0.0, 1.0)}],
        extract_clip_fn=lambda src, s, e: src,
    )
    assert len(segments) == 1


def test_run_pipeline_uses_word_timestamps_and_suppresses_hallucination(monkeypatch):
    # return_timestamps="word": align 단계가 영어 자막 줄 시간에 맞는 한국어를 고를 수
    # 있도록 단어 단위 타임코드를 받는다. 아울러 언어 고정 + 문맥 비조건화 + 무음/반복
    # 억제 임계값으로 환각을 막고, temperature 폴백 시퀀스를 명시해 크래시를 피한다
    # (모두 실측으로 확인된 필수 설정).
    captured = {}

    class FakePipe:
        def __call__(self, audio_path, return_timestamps=None, generate_kwargs=None):
            captured["audio_path"] = audio_path
            captured["return_timestamps"] = return_timestamps
            captured["generate_kwargs"] = generate_kwargs
            return {"chunks": [{"text": "대사", "timestamp": (0.0, 1.0)}]}

    monkeypatch.setattr(local_stt, "_get_pipeline", lambda: FakePipe())
    result = local_stt._run_pipeline("/tmp/full.wav")

    assert result == [{"text": "대사", "timestamp": (0.0, 1.0)}]
    assert captured["return_timestamps"] == "word"
    gk = captured["generate_kwargs"]
    assert gk["language"] == "korean"
    assert gk["task"] == "transcribe"
    assert gk["condition_on_prev_tokens"] is False
    assert "no_speech_threshold" in gk
    assert "logprob_threshold" in gk
    assert "compression_ratio_threshold" in gk
    assert gk["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
