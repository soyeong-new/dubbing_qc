import math
import struct
import wave
from app.core.rule_checks import read_wav_mono, check_audio_quality, _token_similarity
from app.schemas import AlignedPair, SegmentText


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"{len(samples)}h", *samples))


def sine(seconds, rate=16000, amp=8000, freq=440):
    n = int(seconds * rate)
    return [int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]


def pair_at(start, end, kr="대사", en="line", pid="p1"):
    return AlignedPair(
        id=pid,
        korean=SegmentText(start=start, end=end, speaker="A", text=kr),
        dubbed=SegmentText(start=start, end=end, speaker="A", text=en),
    )


def test_read_wav_mono(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, sine(1.0))
    samples, rate = read_wav_mono(str(p))
    assert rate == 16000
    assert len(samples) == 16000


def test_clipping_detected(tmp_path):
    p = tmp_path / "clip.wav"
    write_wav(p, [32760, -32760] * 8000)  # 전부 클리핑
    findings = check_audio_quality(str(p), [pair_at(0.0, 1.0)])
    assert any(f.issue_type == "클리핑" and f.axis == "음질" for f in findings)


def test_dropout_detected_inside_segment(tmp_path):
    p = tmp_path / "drop.wav"
    write_wav(p, sine(1.0) + [0] * 16000 + sine(1.0))  # 1~2초 무음
    findings = check_audio_quality(str(p), [pair_at(0.5, 2.5, pid="p1")])
    assert any(f.issue_type == "드롭아웃" for f in findings)


def test_clean_audio_passes(tmp_path):
    p = tmp_path / "ok.wav"
    # 실제 대사 스템처럼 발화 사이에 무음 휴지가 있는 형태 (무음이 노이즈 플로어 역할)
    write_wav(p, sine(1.0) + [0] * 8000 + sine(1.0))
    findings = check_audio_quality(str(p), [pair_at(0.0, 1.0)])  # 세그먼트는 발화 구간만
    assert findings == []


def test_token_similarity():
    assert _token_similarity("did you eat rice", "did you eat rice") == 1.0
    assert _token_similarity("hello world", "completely different words") < 0.4


def test_check_dialogue_timing_sync_flags_offset_onset(tmp_path):
    from app.core.rule_checks import check_dialogue_timing_sync

    # 원본: 0.5초 무음 후 발화 시작
    kr_clip = tmp_path / "kr_clip.wav"
    write_wav(kr_clip, [0] * 8000 + sine(1.0))
    # 더빙: 1.3초 무음 후 발화 시작 (원본 대비 0.8초 밀림)
    en_clip = tmp_path / "en_clip.wav"
    write_wav(en_clip, [0] * 20800 + sine(1.0))

    def fake_extract(src, start, end):
        return str(kr_clip) if src == "kr_original.wav" else str(en_clip)

    pairs = [pair_at(1.0, 3.0, pid="p1")]
    findings = check_dialogue_timing_sync(
        pairs, "kr_original.wav", "en_stem.wav",
        extract_clip_fn=fake_extract, tolerance=0.5,
    )
    assert len(findings) == 1
    assert findings[0].issue_type == "발화 타이밍 불일치"
    assert findings[0].axis == "싱크 정확도"
    assert findings[0].recommendation.isascii()  # recommendation은 반드시 영어


def test_check_dialogue_timing_sync_passes_when_aligned(tmp_path):
    from app.core.rule_checks import check_dialogue_timing_sync

    # 원본/더빙 모두 0.5초 무음 후 발화 시작 (차이 없음)
    clip = tmp_path / "aligned_clip.wav"
    write_wav(clip, [0] * 8000 + sine(1.0))

    findings = check_dialogue_timing_sync(
        pairs=[pair_at(1.0, 3.0, pid="p1")],
        kr_audio_path="kr.wav", stem_audio_path="en.wav",
        extract_clip_fn=lambda src, s, e: str(clip), tolerance=0.5,
    )
    assert findings == []


def test_check_dialogue_timing_sync_skips_when_either_side_missing():
    from app.core.rule_checks import check_dialogue_timing_sync
    from app.schemas import AlignedPair, SegmentText

    pairs = [AlignedPair(
        id="p1", korean=None,
        dubbed=SegmentText(start=0, end=1, speaker="A", text="hi"),
    )]
    findings = check_dialogue_timing_sync(
        pairs, "kr.wav", "en.wav", extract_clip_fn=lambda s, a, b: s,
    )
    assert findings == []


def test_check_dialogue_timing_sync_skips_when_no_speech_detected(tmp_path):
    from app.core.rule_checks import check_dialogue_timing_sync

    silent_clip = tmp_path / "silent.wav"
    write_wav(silent_clip, [0] * 16000)  # 완전 무음

    findings = check_dialogue_timing_sync(
        pairs=[pair_at(0.0, 1.0, pid="p1")],
        kr_audio_path="kr.wav", stem_audio_path="en.wav",
        extract_clip_fn=lambda src, s, e: str(silent_clip),
    )
    assert findings == []
