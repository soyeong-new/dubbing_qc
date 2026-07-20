from app.schemas import (
    AXES, SegmentText, AlignedPair, QCFinding, AxisScore,
    Verdict, QCJobInput, QCResult, FeedbackEntry,
)


def test_axes_are_the_six_company_axes():
    assert AXES == ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성", "억양 적합성"]


def test_aligned_pair_allows_missing_side():
    pair = AlignedPair(
        id="pair_1",
        korean=SegmentText(start=1.0, end=2.0, speaker="A", text="밥 먹었어?"),
        dubbed=None,
    )
    assert pair.dubbed is None
    assert pair.alignment_confidence == 1.0
    assert pair.scene_id == ""


def test_qcfinding_new_fields_have_defaults():
    f = QCFinding(
        id="f1", segment_id="pair_1", category="localization",
        severity="high", issue_type="번역 오류", start_time=1.0, end_time=2.0,
        speaker="A", description="설명", original_text="원문",
        current_translation="dub", recommendation="Fix it.", confidence=0.9,
    )
    assert f.axis == "언어 적합성"
    assert f.source == "rule"
    assert f.agreement == 1
    assert f.alternatives == {}


def test_verdict_roundtrip():
    v = Verdict(
        status="fail",
        axis_scores=[AxisScore(axis="음질", mos=2, deduction_rate=50.0)],
        reasons=["음질 MOS 2"],
    )
    assert v.status == "fail"


def test_job_input_requires_en_srt_only():
    job = QCJobInput(en_srt_path="/tmp/en.srt")
    assert job.kr_srt_path is None
    assert job.movie_title == "untitled"


def test_feedback_entry_defaults():
    e = FeedbackEntry(
        movie="m", segment_id="pair_1", korean="ㄱ", dubbed="d",
        finding_id="f1", reviewer_action="approved",
    )
    assert e.final_text == ""
    assert e.chosen_persona == ""


def test_axes_has_six_entries_including_accent():
    from app.schemas import AXES
    assert AXES == ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성", "억양 적합성"]


def test_qcfinding_finding_type_defaults_to_quality():
    from app.schemas import QCFinding
    f = QCFinding(
        id="f1", segment_id="p1", category="localization", severity="low",
        issue_type="테스트", start_time=0, end_time=1, speaker="A",
        description="d", original_text="o", current_translation="c",
        recommendation="r", confidence=0.9,
    )
    assert f.finding_type == "quality"


from app.schemas import HeldSegment, JudgeOutput, QCFinding, QCResult, Verdict


def _finding(**kw):
    base = dict(id="f1", segment_id="pair_1", category="localization", severity="low",
                issue_type="t", start_time=0.0, end_time=1.0, speaker="?",
                description="d", original_text="o", current_translation="c",
                recommendation="r", confidence=0.9)
    base.update(kw)
    return QCFinding(**base)


def test_finding_v3_fields_default():
    f = _finding()
    assert f.heard_korean == ""
    assert f.consensus == ""


def test_judge_output_defaults():
    out = JudgeOutput()
    assert out.findings == [] and out.unheard_segment_ids == []


def test_qcresult_held_segments():
    r = QCResult(
        verdict=Verdict(status="pass", axis_scores=[]),
        findings=[], pairs=[],
        held=[HeldSegment(scene_id="scene_1", segment_id="pair_3",
                          start=10.0, end=12.0, reason="청취 불가")],
    )
    assert r.held[0].reason == "청취 불가"
