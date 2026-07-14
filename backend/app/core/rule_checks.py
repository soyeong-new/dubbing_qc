from typing import List
from app.schemas import AlignedPair, QCFinding


def _finding(kind: str, pair: AlignedPair, severity: str, issue_type: str,
             axis: str, description: str, recommendation: str,
             category: str = "localization") -> QCFinding:
    anchor = pair.korean or pair.dubbed
    return QCFinding(
        id=f"rule_{kind}_{pair.id}", segment_id=pair.id, category=category,
        severity=severity, issue_type=issue_type,
        start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
        description=description,
        original_text=pair.korean.text if pair.korean else "",
        current_translation=pair.dubbed.text if pair.dubbed else "",
        recommendation=recommendation, confidence=1.0,
        axis=axis, source="rule",
    )


def check_missing(pairs: List[AlignedPair]) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if p.korean is not None and (p.dubbed is None or not p.dubbed.text.strip()):
            findings.append(_finding(
                "missing", p, "high", "번역 누락", "언어 적합성",
                "해당 한국어 대사에 대응하는 영어 더빙 대사가 없습니다.",
                "Provide the missing dubbed line.",
            ))
    return findings


def check_pacing(pairs: List[AlignedPair], max_words_per_sec: float = 3.8) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if not p.dubbed or not p.dubbed.text.strip():
            continue
        duration = p.dubbed.end - p.dubbed.start
        if duration <= 0:
            continue
        wps = len(p.dubbed.text.split()) / duration
        if wps > max_words_per_sec:
            findings.append(_finding(
                "pacing", p, "medium", "발화속도 초과", "싱크 정확도",
                f"발화속도가 초당 {wps:.1f}단어로 기준({max_words_per_sec})을 초과합니다. "
                "성우 발화가 빨라져 입 싱크가 어긋날 수 있습니다.",
                "Shorten the line to fit the timing.", category="voice",
            ))
    return findings


def check_sync_overflow(pairs: List[AlignedPair], tolerance: float = 0.5) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if not p.korean or not p.dubbed:
            continue
        overflow = max(p.korean.start - p.dubbed.start, p.dubbed.end - p.korean.end)
        if overflow > tolerance:
            findings.append(_finding(
                "sync", p, "medium", "싱크 오버플로", "싱크 정확도",
                f"더빙 구간이 원본 대사 구간을 {overflow:.1f}초 벗어납니다.",
                "Re-time the dubbed line to match the original segment.",
                category="voice",
            ))
    return findings


def check_low_alignment(pairs: List[AlignedPair], min_confidence: float = 0.3) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if p.korean and p.dubbed and p.alignment_confidence < min_confidence:
            findings.append(_finding(
                "lowalign", p, "low", "정렬 신뢰도 저하", "싱크 정확도",
                f"한↔영 세그먼트 정렬 신뢰도가 {p.alignment_confidence:.2f}로 낮습니다. "
                "타임코드 검토가 필요합니다.",
                "Verify the timecode mapping manually.", category="voice",
            ))
    return findings


def run_text_checks(pairs: List[AlignedPair]) -> List[QCFinding]:
    return (check_missing(pairs) + check_pacing(pairs)
            + check_sync_overflow(pairs) + check_low_alignment(pairs))
