import json
from app.providers.gemini import build_judge_prompt, parse_judge_response, parse_stt_response
from app.providers.base import Persona
from app.schemas import AlignedPair, SegmentText

PERSONA = Persona(key="native", name="영어 원어민 시청자",
                  instruction="영어 자연스러움만 평가하라.", axes=["자연스러움"])

PAIR = AlignedPair(
    id="pair_3",
    korean=SegmentText(start=6.0, end=7.5, speaker="민우", text="어이가 없네."),
    dubbed=SegmentText(start=6.1, end=7.4, speaker="민우", text="I have no kidney."),
)


def test_build_judge_prompt_contains_persona_and_dialogue():
    prompt = build_judge_prompt([PAIR], PERSONA, knowledge="- 어이가 없네: ridiculous")
    assert "영어 원어민 시청자" in prompt
    assert "I have no kidney." in prompt
    assert "어이가 없네" in prompt
    assert "한국어" in prompt  # description 한국어 강제 지시


def test_parse_judge_response_maps_to_findings():
    raw = json.dumps([{
        "segment_id": "pair_3", "severity": "high", "issue_type": "번역 오류",
        "description": "치명적 오역입니다.", "recommendation": "This is ridiculous.",
        "confidence": 0.97, "axis": "자연스러움",
    }])
    findings = parse_judge_response(raw, [PAIR], PERSONA)
    assert len(findings) == 1
    f = findings[0]
    assert f.segment_id == "pair_3"
    assert f.source == "persona:native"
    assert f.start_time == 6.0
    assert f.current_translation == "I have no kidney."


def test_parse_judge_response_drops_unknown_segment_and_bad_axis():
    raw = json.dumps([
        {"segment_id": "no_such", "severity": "low", "issue_type": "기타",
         "description": "x", "recommendation": "y", "confidence": 0.5},
        {"segment_id": "pair_3", "severity": "low", "issue_type": "기타",
         "description": "x", "recommendation": "y", "confidence": 0.5,
         "axis": "없는축"},
    ])
    findings = parse_judge_response(raw, [PAIR], PERSONA)
    assert len(findings) == 1
    assert findings[0].axis == "자연스러움"  # 페르소나 기본 축으로 보정


def test_parse_stt_response():
    raw = json.dumps([
        {"start": 1.2, "end": 4.5, "speaker": "인물1", "text": "안녕하세요"},
        {"start": 5.0, "end": 6.0, "speaker": "인물2", "text": "반갑습니다"},
    ])
    segments = parse_stt_response(raw)
    assert len(segments) == 2
    assert segments[0].text == "안녕하세요"
