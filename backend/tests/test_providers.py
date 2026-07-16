import pytest
from app.providers.base import get_provider, ProviderNotConfiguredError, Persona
from app.schemas import AlignedPair, SegmentText


def test_gemini_without_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("QC_PROVIDER", raising=False)
    with pytest.raises(ProviderNotConfiguredError):
        get_provider()


def test_mock_provider_allowed_under_pytest(monkeypatch):
    # pytest 실행 중에는 PYTEST_CURRENT_TEST가 자동 설정된다
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    assert type(provider).__name__ == "MockProvider"


def test_mock_provider_blocked_outside_pytest(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(ProviderNotConfiguredError):
        get_provider()


async def test_mock_judge_flags_kidney(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    persona = Persona(key="culture", name="한국 문화·언어 전문가",
                      instruction="테스트", axes=["언어 적합성"])
    pair = AlignedPair(
        id="pair_1",
        korean=SegmentText(start=0, end=2, speaker="A", text="어이가 없네."),
        dubbed=SegmentText(start=0, end=2, speaker="A", text="I have no kidney."),
    )
    findings = await provider.judge([pair], persona, knowledge="")
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].source == "persona:culture"
    assert findings[0].axis == "언어 적합성"
