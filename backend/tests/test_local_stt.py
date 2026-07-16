import app.core.local_stt as local_stt
from app.core.local_stt import transcribe_korean, _group_words_into_sentences


# ---- _group_words_into_sentences: 단어 → 문장 묶기 ----

def test_group_words_into_sentences_splits_on_sentence_end():
    words = [
        {"text": " 밥", "timestamp": (0.0, 0.4)},
        {"text": " 먹었어?", "timestamp": (0.4, 1.0)},
        {"text": " 어이가", "timestamp": (1.5, 2.0)},
        {"text": " 없네.", "timestamp": (2.0, 2.6)},
    ]
    sents = _group_words_into_sentences(words)
    assert len(sents) == 2
    assert sents[0].text == "밥 먹었어?"
    assert sents[0].start == 0.0 and sents[0].end == 1.0
    assert sents[1].text == "어이가 없네."
    assert sents[1].start == 1.5 and sents[1].end == 2.6


def test_group_words_into_sentences_applies_offset():
    words = [{"text": " 대사.", "timestamp": (0.2, 0.9)}]
    sents = _group_words_into_sentences(words, offset=100.0)
    assert sents[0].start == 100.2 and sents[0].end == 100.9


def test_group_words_into_sentences_flushes_trailing_words_without_punctuation():
    words = [
        {"text": " 끝나지", "timestamp": (0.0, 0.5)},
        {"text": " 않은", "timestamp": (0.5, 1.0)},
    ]
    sents = _group_words_into_sentences(words)
    assert len(sents) == 1
    assert sents[0].text == "끝나지 않은"


def test_group_words_into_sentences_guards_collapsed_word_timestamps():
    # Whisper가 단어 종료 타임을 시작과 같게 붕괴시켜 end가 앞 단어보다 작아질 수 있다.
    words = [
        {"text": " 첫", "timestamp": (0.0, 1.0)},
        {"text": " 단어.", "timestamp": (0.5, 0.5)},
    ]
    sents = _group_words_into_sentences(words)
    assert len(sents) == 1
    assert sents[0].start == 0.0
    assert sents[0].end == 1.0  # max로 단조 증가 보장 (0.5로 뒷걸음치지 않음)


def test_group_words_into_sentences_skips_empty_words():
    words = [
        {"text": "   ", "timestamp": (0.0, 0.3)},
        {"text": " 실제", "timestamp": (0.3, 0.8)},
        {"text": " 대사.", "timestamp": (0.8, 1.5)},
    ]
    sents = _group_words_into_sentences(words)
    assert len(sents) == 1
    assert sents[0].text == "실제 대사."


def test_group_words_into_sentences_handles_open_ended_last_word():
    # Whisper의 마지막 단어는 종료 타임스탬프가 None일 수 있다
    words = [{"text": " 마지막.", "timestamp": (10.0, None)}]
    sents = _group_words_into_sentences(words)
    assert len(sents) == 1
    assert sents[0].start == 10.0
    assert sents[0].end == 10.0  # end가 없으면 start로 대체


# ---- transcribe_korean: 오디오 전체를 한 번에 넣어 단어를 문장으로 묶기 ----

def test_transcribe_korean_calls_transcribe_fn_once_on_whole_file():
    # 무음 기준으로 직접 잘라 여러 번 호출하지 않는다 — 비명/소음 구간만 통째로
    # 떼어 별도 호출하면 반복 환각이 오히려 심해지는 것을 실측으로 확인했기 때문에,
    # 오디오 전체를 한 번에 넣어 파이프라인 자체의 청킹에 맡긴다.
    calls = []

    def fake_transcribe(audio_path):
        calls.append(audio_path)
        return [
            {"text": " 안녕", "timestamp": (3.1, 3.5)},
            {"text": " 하세요.", "timestamp": (3.5, 4.0)},
        ]

    segments = transcribe_korean("/tmp/original.wav", transcribe_fn=fake_transcribe)

    assert calls == ["/tmp/original.wav"]  # 딱 한 번, 원본 경로 그대로
    assert len(segments) == 1
    assert segments[0].text == "안녕 하세요."
    # Whisper가 돌려주는 타임스탬프가 이미 절대 시간이므로 그대로 사용된다.
    assert segments[0].start == 3.1
    assert segments[0].end == 4.0


def test_transcribe_korean_does_not_import_transformers_at_module_load(monkeypatch):
    # transcribe_fn을 주입하면 실제 모델 로드 경로(_get_pipeline)가 전혀 호출되지 않아야 한다
    def boom():
        raise AssertionError("실제 모델을 로드하면 안 된다")

    monkeypatch.setattr(local_stt, "_get_pipeline", boom)
    segments = transcribe_korean(
        "/tmp/x.wav", transcribe_fn=lambda p: [{"text": " ok.", "timestamp": (0.0, 1.0)}],
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
    assert captured["audio_path"] == "/tmp/full.wav"
    assert captured["return_timestamps"] == "word"
    gk = captured["generate_kwargs"]
    assert gk["language"] == "korean"
    assert gk["task"] == "transcribe"
    assert gk["condition_on_prev_tokens"] is False
    assert "no_speech_threshold" in gk
    assert "logprob_threshold" in gk
    assert "compression_ratio_threshold" in gk
    assert gk["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
