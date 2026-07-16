from app.core.local_stt import transcribe_korean


def test_transcribe_korean_uses_injected_fn_and_parses_chunks():
    def fake_transcribe_fn(audio_path):
        assert audio_path == "/tmp/original.wav"
        return [
            {"text": " 형, 밥 먹었어?", "timestamp": (1.0, 3.0)},
            {"text": " 어이가 없네.", "timestamp": (5.2, 7.8)},
        ]

    segments = transcribe_korean("/tmp/original.wav", transcribe_fn=fake_transcribe_fn)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 3.0
    assert segments[0].text == "형, 밥 먹었어?"
    assert segments[1].text == "어이가 없네."


def test_transcribe_korean_skips_empty_chunks():
    def fake_transcribe_fn(audio_path):
        return [
            {"text": "   ", "timestamp": (0.0, 1.0)},
            {"text": "실제 대사", "timestamp": (1.0, 2.0)},
        ]

    segments = transcribe_korean("/tmp/x.wav", transcribe_fn=fake_transcribe_fn)
    assert len(segments) == 1
    assert segments[0].text == "실제 대사"


def test_transcribe_korean_handles_open_ended_last_chunk():
    # Whisper의 마지막 청크는 종료 타임스탬프가 None일 수 있다
    def fake_transcribe_fn(audio_path):
        return [{"text": "마지막 대사", "timestamp": (10.0, None)}]

    segments = transcribe_korean("/tmp/x.wav", transcribe_fn=fake_transcribe_fn)
    assert len(segments) == 1
    assert segments[0].start == 10.0
    assert segments[0].end == 10.0  # end가 없으면 start로 대체


def test_transcribe_korean_does_not_import_transformers_at_module_load(monkeypatch):
    # transcribe_fn을 주입하면 실제 모델 로드 경로(_get_pipeline)가 전혀 호출되지 않아야 한다
    import app.core.local_stt as local_stt

    def boom():
        raise AssertionError("실제 모델을 로드하면 안 된다")

    monkeypatch.setattr(local_stt, "_get_pipeline", boom)
    segments = transcribe_korean(
        "/tmp/x.wav", transcribe_fn=lambda p: [{"text": "ok", "timestamp": (0.0, 1.0)}]
    )
    assert len(segments) == 1
