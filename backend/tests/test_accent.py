from app.core.accent import check_accent_conformance
from app.schemas import AlignedPair, SegmentText


def pair(pid="p1", en_text="line"):
    return AlignedPair(
        id=pid,
        korean=SegmentText(start=0.0, end=2.0, speaker="A", text="대사"),
        dubbed=SegmentText(start=0.0, end=2.0, speaker="A", text=en_text),
    )


def fake_extract_clip(src, start, end):
    return src  # 실제 ffmpeg 호출 없이 원본 경로 반환


def test_check_accent_conformance_flags_non_target_accent():
    def fake_classify(clip_path):
        return "british", 0.9

    findings = check_accent_conformance(
        [pair()], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=fake_classify,
    )
    assert len(findings) == 1
    assert findings[0].axis == "억양 적합성"
    assert findings[0].finding_type == "quality"


def test_check_accent_conformance_passes_target_accent():
    def fake_classify(clip_path):
        return "us", 0.95

    findings = check_accent_conformance(
        [pair()], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=fake_classify,
    )
    assert findings == []


def test_check_accent_conformance_flags_low_confidence_even_if_target_label():
    def fake_classify(clip_path):
        return "us", 0.2  # 라벨은 맞지만 확신도가 낮음

    findings = check_accent_conformance(
        [pair()], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=fake_classify,
        confidence_threshold=0.6,
    )
    assert len(findings) == 1


def test_check_accent_conformance_skips_missing_dub_text():
    findings = check_accent_conformance(
        [pair(en_text="")], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=lambda c: ("us", 0.9),
    )
    assert findings == []
