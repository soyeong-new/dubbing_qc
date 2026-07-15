from app.core.rule_checks import (
    check_missing, check_pacing, check_sync_overflow,
    check_low_alignment, run_text_checks,
)
from app.schemas import AlignedPair, SegmentText


def pair(pid="p1", kr_text="대사", en_text="line", kr=(1.0, 3.0), en=(1.0, 3.0), conf=1.0):
    return AlignedPair(
        id=pid,
        korean=SegmentText(start=kr[0], end=kr[1], speaker="A", text=kr_text) if kr_text is not None else None,
        dubbed=SegmentText(start=en[0], end=en[1], speaker="A", text=en_text) if en_text is not None else None,
        alignment_confidence=conf,
    )


def test_check_missing_flags_empty_dub():
    findings = check_missing([pair(en_text=None, conf=0.0), pair(pid="p2")])
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].issue_type == "번역 누락"
    assert findings[0].axis == "언어 적합성"
    assert findings[0].source == "rule"


def test_check_pacing_flags_fast_speech():
    long_line = " ".join(["word"] * 20)  # 20단어 / 2초 = 10 wps
    findings = check_pacing([pair(en_text=long_line, en=(1.0, 3.0))])
    assert len(findings) == 1
    assert findings[0].axis == "싱크 정확도"
    assert findings[0].issue_type == "발화속도 초과"


def test_check_pacing_passes_normal_speech():
    assert check_pacing([pair(en_text="short line", en=(1.0, 3.0))]) == []


def test_check_sync_overflow():
    findings = check_sync_overflow([pair(kr=(1.0, 3.0), en=(1.0, 4.2))])
    assert len(findings) == 1
    assert findings[0].issue_type == "싱크 오버플로"


def test_check_low_alignment():
    findings = check_low_alignment([pair(conf=0.1)])
    assert len(findings) == 1
    assert findings[0].issue_type == "정렬 신뢰도 저하"


def test_run_text_checks_combines_all():
    pairs = [pair(en_text=None, conf=0.0), pair(pid="p2", kr=(1.0, 3.0), en=(1.0, 4.5))]
    findings = run_text_checks(pairs)
    types = {f.issue_type for f in findings}
    assert "번역 누락" in types
    assert "싱크 오버플로" in types


def test_load_sensitive_terms_reads_yaml(tmp_path):
    from app.core.rule_checks import load_sensitive_terms
    p = tmp_path / "sensitive_words.yaml"
    p.write_text("terms:\n  - word: TESTWORD\n    category: 테스트\n", encoding="utf-8")
    terms = load_sensitive_terms(str(p))
    assert terms == [("testword", "테스트")]


def test_check_sensitive_words_flags_matching_dub_text():
    from app.core.rule_checks import check_sensitive_words
    findings = check_sensitive_words(
        [pair(en_text="this contains TESTWORD in it")],
        terms=[("testword", "테스트")],
    )
    assert len(findings) == 1
    assert findings[0].finding_type == "sensitive"
    assert findings[0].axis == "언어 적합성"
    assert "테스트" in findings[0].description


def test_check_sensitive_words_no_match_returns_empty():
    from app.core.rule_checks import check_sensitive_words
    findings = check_sensitive_words(
        [pair(en_text="a perfectly clean line")],
        terms=[("testword", "테스트")],
    )
    assert findings == []
