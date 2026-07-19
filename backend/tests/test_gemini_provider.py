import json
from app.providers.gemini import build_judge_prompt, parse_judge_response
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
    # 영어 SRT(dubbed)가 타임코드 기준(주체) — 한국어 STT 경계(6.0)가 아니라
    # 더빙 시작 시각(6.1)을 써야 영상 이동·표시가 실제 대사 위치와 맞는다.
    assert f.start_time == 6.1
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


def test_parse_judge_response_reads_finding_type():
    raw = json.dumps([{
        "segment_id": "pair_3", "severity": "high", "issue_type": "민감어",
        "description": "설명", "recommendation": "fix", "confidence": 0.9,
        "axis": "언어 적합성", "finding_type": "sensitive",
    }])
    findings = parse_judge_response(raw, [PAIR], PERSONA)
    assert findings[0].finding_type == "sensitive"


def test_parse_judge_response_defaults_finding_type_to_quality():
    raw = json.dumps([{
        "segment_id": "pair_3", "severity": "low", "issue_type": "기타",
        "description": "d", "recommendation": "r", "confidence": 0.5,
    }])
    findings = parse_judge_response(raw, [PAIR], PERSONA)
    assert findings[0].finding_type == "quality"


def test_parse_judge_response_rejects_invalid_finding_type():
    raw = json.dumps([{
        "segment_id": "pair_3", "severity": "low", "issue_type": "기타",
        "description": "d", "recommendation": "r", "confidence": 0.5,
        "finding_type": "not_a_real_type",
    }])
    findings = parse_judge_response(raw, [PAIR], PERSONA)
    assert findings[0].finding_type == "quality"


def test_judge_attaches_both_original_and_dub_audio(monkeypatch, tmp_path):
    import asyncio
    from unittest.mock import MagicMock
    from app.providers.gemini import GeminiProvider

    # uses_audio=True인 페르소나(연출가 등)여야 오디오 파트가 실제로 붙는다.
    # 모듈 상단의 PERSONA(native)는 uses_audio=False라 이 케이스를 검증하지 못한다.
    audio_persona = Persona(
        key="director", name="더빙 연출가",
        instruction="감정 표현을 평가하라.", axes=["감정 표현"], uses_audio=True,
    )

    dub_wav = tmp_path / "dub.wav"
    orig_wav = tmp_path / "orig.wav"
    dub_wav.write_bytes(b"fake")
    orig_wav.write_bytes(b"fake")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.providers.gemini._compress_to_mp3", lambda path: b"mp3data")

    captured = {}

    class FakeModel:
        def generate_content(self, parts, generation_config=None):
            captured["parts"] = parts
            resp = MagicMock()
            resp.text = "[]"
            return resp

    class FakeGenAI:
        def configure(self, api_key=None):
            pass

        def GenerativeModel(self, name):
            return FakeModel()

    monkeypatch.setattr("google.generativeai.configure", lambda api_key=None: None)
    provider = GeminiProvider()
    provider._genai = FakeGenAI()

    asyncio.run(provider.judge(
        [PAIR], audio_persona, knowledge="",
        audio_clip_path=str(dub_wav), original_audio_clip_path=str(orig_wav),
    ))
    # 오디오 파트가 2개(원본+더빙) 포함되어야 한다
    audio_parts = [p for p in captured["parts"] if isinstance(p, dict) and "mime_type" in p]
    assert len(audio_parts) == 2
