from app.core.verdict import load_config, compute_axis_scores, decide
from app.schemas import QCFinding, AXES


def finding(axis, severity, seg="p1"):
    return QCFinding(
        id=f"f_{axis}_{severity}_{seg}", segment_id=seg, category="localization",
        severity=severity, issue_type="번역 오류", start_time=0, end_time=1,
        speaker="A", description="d", original_text="o",
        current_translation="c", recommendation="r", confidence=0.9, axis=axis,
    )


def test_all_axes_always_scored():
    config = load_config()
    scores = compute_axis_scores([], n_pairs=100, config=config)
    assert [s.axis for s in scores] == AXES
    assert all(s.mos == 5 for s in scores)


def test_deductions_lower_mos():
    config = load_config()
    findings = [finding("언어 적합성", "medium", seg=f"p{i}") for i in range(10)]
    scores = compute_axis_scores(findings, n_pairs=100, config=config)
    lang = next(s for s in scores if s.axis == "언어 적합성")
    assert lang.mos < 5
    others = [s for s in scores if s.axis != "언어 적합성"]
    assert all(s.mos == 5 for s in others)


def test_pass_when_all_axes_4_or_above():
    config = load_config()
    scores = compute_axis_scores([], n_pairs=100, config=config)
    verdict = decide(scores, [], config)
    assert verdict.status == "pass"


def test_single_high_finding_forces_fail():
    config = load_config()
    findings = [finding("언어 적합성", "high")]
    scores = compute_axis_scores(findings, n_pairs=1000, config=config)
    verdict = decide(scores, findings, config)
    assert verdict.status == "fail"
    assert any("high" in r or "치명" in r for r in verdict.reasons)


def test_conditional_when_one_axis_is_3():
    config = load_config()
    # medium 25건/100세그 → 감점률 200 → 해당 축 MOS 낮음. 정확한 경계는 config 기준으로 계산:
    # deduction_rate = 25*8 = 200/100pairs = 200.0 → mos 1. 대신 적은 수로 3 유도:
    findings = [finding("자연스러움", "medium", seg=f"p{i}") for i in range(3)]
    scores = compute_axis_scores(findings, n_pairs=100, config=config)
    nat = next(s for s in scores if s.axis == "자연스러움")
    assert nat.deduction_rate == 24.0  # 3건 * 8점 / 100세그 * 100
    assert nat.mos == 3               # 15 < 24 <= 35 구간
    verdict = decide(scores, findings, config)
    assert verdict.status == "conditional"
