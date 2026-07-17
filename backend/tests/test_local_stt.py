import app.core.local_stt as local_stt
from app.core.local_stt import transcribe_korean


# ---- transcribe_korean: 오디오 전체를 한 번에 넣어 단어 그대로 반환 ----

def test_transcribe_korean_calls_transcribe_fn_once_on_whole_file():
    # 무음 기준으로 직접 잘라 여러 번 호출하지 않는다 — 비명/소음 구간만 통째로
    # 떼어 별도 호출하면 반복 환각이 오히려 심해지는 것을 실측으로 확인했기 때문에,
    # 오디오 전체를 한 번에 넣어 파이프라인 자체의 청킹에 맡긴다.
    calls = []

    def fake_transcribe(audio_path):
        calls.append(audio_path)
        return [
            {"text": " 안녕", "timestamp": (3.1, 3.5)},
            {"text": " 하세요", "timestamp": (3.5, 4.0)},
        ]

    segments = transcribe_korean("/tmp/original.wav", transcribe_fn=fake_transcribe)

    assert calls == ["/tmp/original.wav"]  # 딱 한 번, 원본 경로 그대로
    assert len(segments) == 2  # 문장으로 미리 묶지 않고 단어 그대로 반환
    assert segments[0].text == "안녕"
    assert segments[0].start == 3.1 and segments[0].end == 3.5
    assert segments[1].text == "하세요"
    assert segments[1].start == 3.5 and segments[1].end == 4.0


def test_transcribe_korean_skips_empty_words():
    def fake_transcribe(audio_path):
        return [
            {"text": "   ", "timestamp": (0.0, 0.3)},
            {"text": " 실제", "timestamp": (0.3, 0.8)},
        ]

    segments = transcribe_korean("/tmp/x.wav", transcribe_fn=fake_transcribe)
    assert len(segments) == 1
    assert segments[0].text == "실제"


def test_transcribe_korean_handles_open_ended_last_word():
    # Whisper의 마지막 단어는 종료 타임스탬프가 None일 수 있다
    def fake_transcribe(audio_path):
        return [{"text": " 마지막", "timestamp": (10.0, None)}]

    segments = transcribe_korean("/tmp/x.wav", transcribe_fn=fake_transcribe)
    assert len(segments) == 1
    assert segments[0].start == 10.0
    assert segments[0].end == 10.0  # end가 없으면 start로 대체


def test_transcribe_korean_does_not_import_transformers_at_module_load(monkeypatch):
    # transcribe_fn을 주입하면 실제 모델 로드 경로(_get_pipeline)가 전혀 호출되지 않아야 한다
    def boom():
        raise AssertionError("실제 모델을 로드하면 안 된다")

    monkeypatch.setattr(local_stt, "_get_pipeline", boom)
    segments = transcribe_korean(
        "/tmp/x.wav", transcribe_fn=lambda p: [{"text": " ok", "timestamp": (0.0, 1.0)}],
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
