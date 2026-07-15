from app.core.judge_panel import PERSONAS, merge_findings, run_panel
from app.providers.base import get_provider
from app.schemas import AlignedPair, SegmentText, QCFinding


def make_finding(seg_id, persona_key, severity, rec):
    return QCFinding(
        id=f"{persona_key}_{seg_id}_0", segment_id=seg_id, category="localization",
        severity=severity, issue_type="번역 오류", start_time=0, end_time=1,
        speaker="A", description="설명", original_text="원문",
        current_translation="dub", recommendation=rec, confidence=0.9,
        axis="언어 적합성", source=f"persona:{persona_key}",
    )


def test_personas_are_three_with_director_audio():
    keys = [p.key for p in PERSONAS]
    assert keys == ["culture", "native", "director"]
    assert PERSONAS[2].uses_audio is True
    assert PERSONAS[0].uses_audio is False


def test_merge_upgrades_agreement_and_collects_alternatives():
    merged = merge_findings([
        make_finding("p1", "culture", "medium", "Fix A"),
        make_finding("p1", "native", "high", "Fix B"),
        make_finding("p2", "culture", "low", "Fix C"),
    ])
    by_seg = {f.segment_id: f for f in merged}
    assert by_seg["p1"].agreement == 2
    assert by_seg["p1"].severity == "high"
    assert set(by_seg["p1"].alternatives.values()) == {"Fix A", "Fix B"}
    assert by_seg["p2"].agreement == 1


async def test_run_panel_end_to_end_with_mock(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    pair = AlignedPair(
        id="pair_1", scene_id="scene_1",
        korean=SegmentText(start=0, end=2, speaker="A", text="어이가 없네."),
        dubbed=SegmentText(start=0, end=2, speaker="A", text="I have no kidney."),
    )
    progress = []
    findings = await run_panel(
        {"scene_1": [pair]}, knowledge="", provider=provider,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    # 3개 페르소나 모두 같은 오류를 지적 → 병합 후 1건, agreement 3
    assert len(findings) == 1
    assert findings[0].agreement == 3
    assert progress[-1] == (1, 1)


async def test_run_panel_survives_audio_clip_extraction_failure(monkeypatch):
    # stem_wav_path가 존재하지 않아 ffmpeg 추출이 실패해도 패널 전체가 죽으면 안 된다
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    pair = AlignedPair(
        id="pair_1", scene_id="scene_1",
        korean=SegmentText(start=0, end=2, speaker="A", text="어이가 없네."),
        dubbed=SegmentText(start=0, end=2, speaker="A", text="I have no kidney."),
    )
    findings = await run_panel(
        {"scene_1": [pair]}, knowledge="", provider=provider,
        stem_wav_path="/nonexistent/stem.wav",
    )
    assert len(findings) == 1
    assert findings[0].agreement == 3


def test_native_persona_instruction_mentions_sensitive_content():
    native = next(p for p in PERSONAS if p.key == "native")
    assert "민감" in native.instruction
    assert "finding_type" in native.instruction
