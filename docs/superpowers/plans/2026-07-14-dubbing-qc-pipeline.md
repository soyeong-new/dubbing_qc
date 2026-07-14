# 더빙 QC 파이프라인 재설계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 더빙 완성본(영어 SRT + 다이얼로그 스템)을 한국어 원본과 대조해 5축 MOS 판정과 수정 지시서를 내는 실제 QC 파이프라인 구축 (스펙: `docs/superpowers/specs/2026-07-14-dubbing-qc-redesign-design.md`)

**Architecture:** SRT 파싱/STT → 타임코드 정렬 → 결정론적 룰 체크 → 3-페르소나 LLM Judge 패널 + 병합 → 축별 MOS 판정. 모델 호출은 전부 `providers/` 계약 뒤에 격리. 검수자 행위는 JSONL 피드백 저장소에 축적.

**Tech Stack:** Python 3.14 / FastAPI / Pydantic v2 / google-generativeai (gemini-3.5-flash) / ffmpeg / React 19 + Vite

## Global Constraints

- **mock 자동 폴백 금지.** 운영 경로에서 목 데이터가 결과로 나오면 안 된다. API 키 부재 시 파이프라인은 시작을 거부하고 에러를 반환한다. `MockProvider`는 pytest 실행 중에만 선택 가능.
- QCFinding의 `description`은 반드시 한국어, `recommendation`은 반드시 영어.
- 사내 5축은 정확히 이 문자열: `"음질"`, `"감정 표현"`, `"싱크 정확도"`, `"자연스러움"`, `"언어 적합성"`.
- 판정 상태 문자열: `"pass"` | `"conditional"` | `"fail"`.
- Gemini 모델 ID는 `'gemini-3.5-flash'` (기존 코드와 동일).
- 새 런타임 의존성은 `pyyaml`만 추가. 개발 의존성은 `pytest`, `pytest-asyncio`, `httpx`.
- 백엔드 테스트는 `backend/` 디렉토리에서 `venv/bin/python -m pytest`로 실행 (venv은 `backend/venv`).
- Python 3.14: `audioop` 모듈 없음 — WAV 분석은 `wave`+`struct`로 직접 구현.
- 판정 임계값·감점 가중치는 `backend/app/qc_config.yaml`에서 로드 (코드 하드코딩 금지).
- 기존 파일 중 `context.py`, `voice_qc.py`는 삭제하지 않고 유지한다 (v1 파이프라인에서는 호출하지 않음 — 시각 맥락/음색 검사는 실모델 연동 시 재활성화).

---

### Task 1: 테스트 인프라 + 데이터 계약 (schemas)

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`
- Create: `backend/pytest.ini`
- Create: `backend/tests/__init__.py`
- Modify: `backend/app/schemas.py`
- Test: `backend/tests/test_schemas.py`

**Interfaces:**
- Produces (이후 모든 태스크가 사용):
  - `AXES: list[str]` — 5축 문자열 목록
  - `SegmentText(start: float, end: float, speaker: str = "?", text: str)`
  - `AlignedPair(id: str, korean: SegmentText | None, dubbed: SegmentText | None, scene_id: str = "", alignment_confidence: float = 1.0)`
  - `QCFinding` — 기존 필드 유지 + `axis: str`, `source: str`, `agreement: int`, `alternatives: dict[str, str]` 추가
  - `AxisScore(axis: str, mos: int, deduction_rate: float)`
  - `Verdict(status: str, axis_scores: list[AxisScore], reasons: list[str])`
  - `QCJobInput(movie_title, en_srt_path, kr_srt_path?, kr_audio_path?, stem_audio_path?)`
  - `QCResult(verdict: Verdict, findings: list[QCFinding], pairs: list[AlignedPair])`
  - `FeedbackEntry(movie, segment_id, korean, dubbed, finding_id, reviewer_action, final_text, chosen_persona, timestamp)`

- [ ] **Step 1: 의존성 및 pytest 설정 추가**

`backend/requirements.txt`에 한 줄 추가:

```text
pyyaml>=6.0
```

`backend/requirements-dev.txt` 생성:

```text
pytest>=8.0
pytest-asyncio>=0.24
httpx>=0.27
```

`backend/pytest.ini` 생성:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

`backend/tests/__init__.py` 생성 (빈 파일).

설치: `cd backend && venv/bin/pip install -r requirements.txt -r requirements-dev.txt`

- [ ] **Step 2: 실패하는 테스트 작성**

`backend/tests/test_schemas.py`:

```python
from app.schemas import (
    AXES, SegmentText, AlignedPair, QCFinding, AxisScore,
    Verdict, QCJobInput, QCResult, FeedbackEntry,
)


def test_axes_are_the_five_company_axes():
    assert AXES == ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성"]


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
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ImportError: cannot import name 'AXES'`

- [ ] **Step 4: schemas.py 확장 구현**

`backend/app/schemas.py`의 기존 `QCFinding` 클래스를 아래로 교체하고, 파일 끝에 신규 모델들을 추가:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

AXES = ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성"]


class SegmentText(BaseModel):
    start: float
    end: float
    speaker: str = "?"
    text: str


class AlignedPair(BaseModel):
    id: str
    korean: Optional[SegmentText] = None
    dubbed: Optional[SegmentText] = None
    scene_id: str = ""
    alignment_confidence: float = 1.0


class QCFinding(BaseModel):
    id: str
    segment_id: str
    category: str = Field(..., description="'localization' or 'voice'")
    severity: str = Field(..., description="'high' | 'medium' | 'low'")
    issue_type: str
    start_time: float
    end_time: float
    speaker: str
    description: str = Field(..., description="반드시 한국어")
    original_text: str
    current_translation: str
    recommendation: str = Field(..., description="반드시 영어 더빙 대사")
    confidence: float
    axis: str = "언어 적합성"
    source: str = "rule"
    agreement: int = 1
    alternatives: Dict[str, str] = Field(default_factory=dict)


class AxisScore(BaseModel):
    axis: str
    mos: int = Field(..., ge=1, le=5)
    deduction_rate: float


class Verdict(BaseModel):
    status: Literal["pass", "conditional", "fail"]
    axis_scores: List[AxisScore]
    reasons: List[str] = Field(default_factory=list)


class QCJobInput(BaseModel):
    movie_title: str = "untitled"
    en_srt_path: str
    kr_srt_path: Optional[str] = None
    kr_audio_path: Optional[str] = None
    stem_audio_path: Optional[str] = None


class QCResult(BaseModel):
    verdict: Verdict
    findings: List[QCFinding]
    pairs: List[AlignedPair]


class FeedbackEntry(BaseModel):
    movie: str
    segment_id: str
    korean: str
    dubbed: str
    finding_id: str
    reviewer_action: Literal["approved", "rejected", "modified"]
    final_text: str = ""
    chosen_persona: str = ""
    timestamp: str = ""
```

기존 `ScriptSegment`, `QCRequest`, `QCStats`, `QCResponse`는 그대로 둔다 (기존 엔드포인트가 아직 참조 — Task 12에서 정리).

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_schemas.py -v`
Expected: 6 passed

- [ ] **Step 6: 커밋**

```bash
git add backend/requirements.txt backend/requirements-dev.txt backend/pytest.ini backend/tests backend/app/schemas.py
git commit -m "feat: QC 데이터 계약 확장 (AlignedPair, 5축 QCFinding, Verdict) + pytest 인프라"
```

---

### Task 2: 프로바이더 계약 + 팩토리 (mock 자동 폴백 제거)

**Files:**
- Create: `backend/app/providers/__init__.py`
- Create: `backend/app/providers/base.py`
- Create: `backend/app/providers/mock.py`
- Test: `backend/tests/test_providers.py`

**Interfaces:**
- Consumes: Task 1의 `SegmentText`, `AlignedPair`, `QCFinding`
- Produces:
  - `Persona(key: str, name: str, instruction: str, uses_audio: bool = False, axes: list[str])` (pydantic)
  - `ProviderNotConfiguredError(RuntimeError)`
  - `ModelProvider` (ABC): `async transcribe(audio_path: str, lang: str) -> list[SegmentText]`, `async judge(pairs: list[AlignedPair], persona: Persona, knowledge: str, audio_clip_path: str | None = None) -> list[QCFinding]`
  - `get_provider() -> ModelProvider` — `QC_PROVIDER=mock`은 pytest 중에만 허용, Gemini는 `GEMINI_API_KEY` 없으면 raise
  - `MockProvider` — 테스트 전용 결정론적 구현

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_providers.py`:

```python
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


async def test_mock_transcribe_returns_segments(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    segments = await provider.transcribe("/tmp/nonexistent.wav", lang="ko")
    assert len(segments) >= 2
    assert segments[0].start < segments[1].start
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.providers'`

- [ ] **Step 3: 구현**

`backend/app/providers/__init__.py`: 빈 파일.

`backend/app/providers/base.py`:

```python
import os
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas import SegmentText, AlignedPair, QCFinding


class ProviderNotConfiguredError(RuntimeError):
    pass


class Persona(BaseModel):
    key: str
    name: str
    instruction: str
    uses_audio: bool = False
    axes: List[str] = Field(default_factory=list)


class ModelProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        ...

    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        ...

    # score(pairs) -> list[float] : 2단계 xCOMET 스크리너용 예약. 이번 단계에서 구현하지 않음.


def get_provider() -> ModelProvider:
    name = os.getenv("QC_PROVIDER", "gemini")
    if name == "mock":
        # 검수 결과에 목 데이터가 섞이면 안 된다 — 테스트 프로세스에서만 허용
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise ProviderNotConfiguredError("mock 프로바이더는 자동화 테스트 전용입니다.")
        from app.providers.mock import MockProvider
        return MockProvider()
    if name == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise ProviderNotConfiguredError(
                "GEMINI_API_KEY가 설정되지 않았습니다. 검수를 시작할 수 없습니다. "
                ".env 파일 또는 환경변수에 키를 설정하세요."
            )
        from app.providers.gemini import GeminiProvider
        return GeminiProvider()
    raise ProviderNotConfiguredError(f"알 수 없는 프로바이더: {name}")
```

`backend/app/providers/mock.py`:

```python
from typing import List, Optional
from app.providers.base import ModelProvider, Persona
from app.schemas import SegmentText, AlignedPair, QCFinding

# 결정론적 테스트 더블. 운영 경로에서는 base.get_provider()가 선택을 차단한다.
_BAD_PATTERNS = [
    ("kidney", "번역 오류", "high", "관용구 '어이가 없네'가 신장(kidney)으로 오역되었습니다.", "This is ridiculous."),
    ("eat rice", "문화적 정서 차이", "medium", "'밥 먹었어?'가 직역되어 안부 인사의 의미가 사라졌습니다.", "Have you eaten?"),
    ("brother", "문화적 정서 차이", "medium", "호칭 '형'이 brother로 직역되어 어색합니다.", "Hey, man."),
]


class MockProvider(ModelProvider):
    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        return [
            SegmentText(start=1.0, end=4.5, speaker="화자1", text="임마, 너 어제 눈치 보며 기어 다녔다며?"),
            SegmentText(start=5.2, end=7.8, speaker="화자2", text="어이가 없네. 밥도 못 먹고 조사받고 있어요."),
        ]

    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        findings = []
        for pair in pairs:
            if not pair.dubbed or not pair.korean:
                continue
            text_en = pair.dubbed.text.lower()
            for pattern, issue_type, severity, desc, rec in _BAD_PATTERNS:
                if pattern in text_en:
                    findings.append(QCFinding(
                        id=f"{persona.key}_{pair.id}_{pattern.replace(' ', '_')}",
                        segment_id=pair.id, category="localization",
                        severity=severity, issue_type=issue_type,
                        start_time=pair.korean.start, end_time=pair.korean.end,
                        speaker=pair.korean.speaker, description=desc,
                        original_text=pair.korean.text,
                        current_translation=pair.dubbed.text,
                        recommendation=rec, confidence=0.9,
                        axis=persona.axes[0] if persona.axes else "언어 적합성",
                        source=f"persona:{persona.key}",
                    ))
                    break
        return findings
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_providers.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/providers backend/tests/test_providers.py
git commit -m "feat: 모델 프로바이더 계약 + 팩토리 (mock 자동 폴백 차단)"
```

---

### Task 3: Gemini 프로바이더

**Files:**
- Create: `backend/app/providers/gemini.py`
- Test: `backend/tests/test_gemini_provider.py`

**Interfaces:**
- Consumes: Task 2의 `ModelProvider`, `Persona`; Task 1의 스키마
- Produces:
  - `GeminiProvider(ModelProvider)` — `transcribe()`(기존 main.py STT 로직 이관: WAV→24k MP3 압축 후 멀티모달 호출), `judge()`(페르소나 프롬프트 + JSON 응답)
  - 순수 함수 `build_judge_prompt(pairs, persona, knowledge) -> str`
  - 순수 함수 `parse_judge_response(text: str, pairs: list[AlignedPair], persona: Persona) -> list[QCFinding]`
  - 순수 함수 `parse_stt_response(text: str) -> list[SegmentText]`

- [ ] **Step 1: 실패하는 테스트 작성** (LLM 호출 없이 순수 함수만 검증)

`backend/tests/test_gemini_provider.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py -v`
Expected: FAIL — `ModuleNotFoundError` 또는 `ImportError`

- [ ] **Step 3: 구현**

`backend/app/providers/gemini.py`:

```python
import os
import json
import subprocess
import tempfile
from typing import List, Optional
from app.providers.base import ModelProvider, Persona
from app.schemas import SegmentText, AlignedPair, QCFinding, AXES

MODEL_NAME = "gemini-3.5-flash"

STT_PROMPT = """
제공된 {lang_name} 오디오 파일을 듣고, 화자별 대사와 시작/종료 시간(초)을 추출하십시오.
세그먼트는 발화 단위로 1~4초 내외로 분할하십시오.
반드시 아래 JSON 배열만 반환하십시오:
[{{"start": 1.2, "end": 4.5, "speaker": "인물 1", "text": "대사 내용"}}]
"""

JUDGE_PROMPT_TEMPLATE = """
당신은 한국 영화의 영어 더빙을 검수하는 "{persona_name}"입니다.

{persona_instruction}

## 검수 지식베이스 (참고 규칙)
{knowledge}

## 지시
아래 세그먼트 쌍(한국어 원문 ↔ 영어 더빙)을 검토하여 문제가 있는 항목만 JSON 배열로 반환하십시오.
- description은 반드시 **한국어**로: 무엇이 왜 문제인지 설명
- recommendation은 반드시 **영어**로: 교체 가능한 최종 더빙 대사
- axis는 다음 중 하나: {axes}
- severity는 "high" | "medium" | "low"
- 문제 없는 세그먼트는 결과에 포함하지 마십시오.

반환 스키마:
[{{"segment_id": "...", "severity": "...", "issue_type": "...",
  "description": "...", "recommendation": "...", "confidence": 0.9, "axis": "..."}}]

## 분석할 세그먼트 쌍
{payload}
"""


def build_judge_prompt(pairs: List[AlignedPair], persona: Persona, knowledge: str) -> str:
    payload = []
    for p in pairs:
        payload.append({
            "segment_id": p.id,
            "korean": p.korean.text if p.korean else "",
            "english_dub": p.dubbed.text if p.dubbed else "",
            "speaker": p.korean.speaker if p.korean else (p.dubbed.speaker if p.dubbed else "?"),
            "start": p.korean.start if p.korean else (p.dubbed.start if p.dubbed else 0),
            "end": p.korean.end if p.korean else (p.dubbed.end if p.dubbed else 0),
        })
    return JUDGE_PROMPT_TEMPLATE.format(
        persona_name=persona.name,
        persona_instruction=persona.instruction,
        knowledge=knowledge or "(등록된 규칙 없음)",
        axes=" | ".join(persona.axes or AXES),
        payload=json.dumps(payload, ensure_ascii=False, indent=1),
    )


def parse_judge_response(text: str, pairs: List[AlignedPair], persona: Persona) -> List[QCFinding]:
    by_id = {p.id: p for p in pairs}
    default_axis = persona.axes[0] if persona.axes else "언어 적합성"
    findings = []
    for i, item in enumerate(json.loads(text)):
        pair = by_id.get(item.get("segment_id"))
        if pair is None:
            continue
        axis = item.get("axis", default_axis)
        if axis not in AXES:
            axis = default_axis
        anchor = pair.korean or pair.dubbed
        findings.append(QCFinding(
            id=f"{persona.key}_{pair.id}_{i}",
            segment_id=pair.id,
            category="localization",
            severity=item.get("severity", "medium"),
            issue_type=item.get("issue_type", "번역 오류"),
            start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
            description=item.get("description", ""),
            original_text=pair.korean.text if pair.korean else "",
            current_translation=pair.dubbed.text if pair.dubbed else "",
            recommendation=item.get("recommendation", ""),
            confidence=float(item.get("confidence", 0.8)),
            axis=axis,
            source=f"persona:{persona.key}",
        ))
    return findings


def parse_stt_response(text: str) -> List[SegmentText]:
    segments = []
    for item in json.loads(text):
        segments.append(SegmentText(
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            speaker=item.get("speaker", "?"),
            text=item.get("text", ""),
        ))
    return segments


def _compress_to_mp3(audio_path: str) -> bytes:
    out = os.path.join(tempfile.gettempdir(), f"qc_compress_{os.getpid()}.mp3")
    subprocess.run(
        ["ffmpeg", "-i", audio_path, "-acodec", "libmp3lame",
         "-b:a", "24k", "-ar", "16000", "-ac", "1", "-y", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    with open(out, "rb") as f:
        data = f.read()
    os.remove(out)
    return data


class GeminiProvider(ModelProvider):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._genai = genai

    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        lang_name = "한국어" if lang == "ko" else "영어"
        audio_data = _compress_to_mp3(audio_path)
        model = self._genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [{"mime_type": "audio/mp3", "data": audio_data},
             STT_PROMPT.format(lang_name=lang_name)],
            generation_config={"response_mime_type": "application/json"},
        )
        return parse_stt_response(response.text)

    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        model = self._genai.GenerativeModel(MODEL_NAME)
        prompt = build_judge_prompt(pairs, persona, knowledge)
        parts = [prompt]
        if audio_clip_path and persona.uses_audio and os.path.exists(audio_clip_path):
            parts.insert(0, {"mime_type": "audio/mp3", "data": _compress_to_mp3(audio_clip_path)})
        response = model.generate_content(
            parts, generation_config={"response_mime_type": "application/json"},
        )
        return parse_judge_response(response.text, pairs, persona)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/providers/gemini.py backend/tests/test_gemini_provider.py
git commit -m "feat: Gemini 프로바이더 (STT 이관 + 페르소나 judge)"
```

---

### Task 4: SRT 파싱 + 텍스트 출처 결정 (ingest)

**Files:**
- Create: `backend/app/core/ingest.py`
- Test: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `SegmentText`, `ModelProvider`
- Produces:
  - `parse_srt(content: str) -> list[SegmentText]`
  - `async load_text_source(lang: str, srt_path: str | None, audio_path: str | None, provider: ModelProvider) -> list[SegmentText]` — SRT 우선, 없으면 STT 폴백, 둘 다 없으면 `ValueError`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_ingest.py`:

```python
import pytest
from app.core.ingest import parse_srt, load_text_source
from app.providers.base import get_provider

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,500
Hey man, did you eat rice?

2
00:00:05,200 --> 00:00:07,800
This is ridiculous.
It really is.

"""


def test_parse_srt_basic():
    segments = parse_srt(SAMPLE_SRT)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 4.5
    assert segments[0].text == "Hey man, did you eat rice?"
    assert segments[1].text == "This is ridiculous. It really is."


def test_parse_srt_skips_malformed_blocks():
    segments = parse_srt("garbage\n\n1\n00:00:01,000 --> 00:00:02,000\nok\n")
    assert len(segments) == 1
    assert segments[0].text == "ok"


async def test_load_text_source_prefers_srt(tmp_path, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    srt = tmp_path / "en.srt"
    srt.write_text(SAMPLE_SRT, encoding="utf-8")
    segments = await load_text_source("en", str(srt), "/tmp/audio.wav", get_provider())
    assert segments[0].text == "Hey man, did you eat rice?"  # STT가 아닌 SRT 결과


async def test_load_text_source_falls_back_to_stt(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    segments = await load_text_source("ko", None, "/tmp/audio.wav", get_provider())
    assert "눈치" in segments[0].text  # MockProvider.transcribe 결과


async def test_load_text_source_requires_some_input(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    with pytest.raises(ValueError):
        await load_text_source("ko", None, None, get_provider())
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.ingest'`

- [ ] **Step 3: 구현**

`backend/app/core/ingest.py`:

```python
import re
from typing import List, Optional
from app.schemas import SegmentText
from app.providers.base import ModelProvider

_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(content: str) -> List[SegmentText]:
    segments = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        time_idx = next((i for i, ln in enumerate(lines) if _TIME_RE.search(ln)), None)
        if time_idx is None:
            continue
        m = _TIME_RE.search(lines[time_idx])
        text = " ".join(lines[time_idx + 1:]).strip()
        if not text:
            continue
        segments.append(SegmentText(
            start=_to_seconds(*m.groups()[0:4]),
            end=_to_seconds(*m.groups()[4:8]),
            text=text,
        ))
    return segments


async def load_text_source(lang: str, srt_path: Optional[str],
                           audio_path: Optional[str],
                           provider: ModelProvider) -> List[SegmentText]:
    if srt_path:
        with open(srt_path, encoding="utf-8-sig") as f:
            return parse_srt(f.read())
    if audio_path:
        return await provider.transcribe(audio_path, lang)
    raise ValueError(f"{lang}: SRT 또는 오디오 중 하나는 제공되어야 합니다.")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/ingest.py backend/tests/test_ingest.py
git commit -m "feat: SRT 파싱 + 텍스트 출처 결정 (SRT 우선, STT 폴백)"
```

---

### Task 5: 타임코드 정렬 + 씬 배정 (alignment)

**Files:**
- Create: `backend/app/core/alignment.py`
- Test: `backend/tests/test_alignment.py`

**Interfaces:**
- Consumes: `SegmentText`, `AlignedPair`
- Produces:
  - `align(korean: list[SegmentText], dubbed: list[SegmentText]) -> list[AlignedPair]` — 시간 겹침 최대 매칭, `alignment_confidence = 겹침시간 / 합집합시간`, 미매칭 측은 `None`
  - `assign_scenes(pairs: list[AlignedPair], gap_threshold: float = 3.0) -> list[AlignedPair]` — 직전 세그먼트와의 간격이 threshold 초과 시 새 씬, `scene_id = "scene_1"...`
  - `group_by_scene(pairs) -> dict[str, list[AlignedPair]]`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_alignment.py`:

```python
from app.core.alignment import align, assign_scenes, group_by_scene
from app.schemas import SegmentText


def kr(s, e, t):
    return SegmentText(start=s, end=e, speaker="K", text=t)


def en(s, e, t):
    return SegmentText(start=s, end=e, speaker="E", text=t)


def test_align_matches_by_overlap():
    pairs = align(
        korean=[kr(1.0, 3.0, "밥 먹었어?"), kr(4.0, 6.0, "어이가 없네")],
        dubbed=[en(1.1, 3.2, "Did you eat rice?"), en(4.2, 6.1, "I have no kidney")],
    )
    assert len(pairs) == 2
    assert pairs[0].korean.text == "밥 먹었어?"
    assert pairs[0].dubbed.text == "Did you eat rice?"
    assert 0.8 < pairs[0].alignment_confidence <= 1.0


def test_align_reports_unmatched_korean():
    pairs = align(korean=[kr(1.0, 3.0, "대사"), kr(10.0, 12.0, "누락된 대사")],
                  dubbed=[en(1.0, 3.0, "line")])
    assert pairs[1].dubbed is None
    assert pairs[1].alignment_confidence == 0.0


def test_align_reports_extra_dubbed():
    pairs = align(korean=[kr(1.0, 3.0, "대사")],
                  dubbed=[en(1.0, 3.0, "line"), en(20.0, 22.0, "ad-lib")])
    extras = [p for p in pairs if p.korean is None]
    assert len(extras) == 1
    assert extras[0].dubbed.text == "ad-lib"


def test_assign_scenes_by_gap():
    pairs = align(
        korean=[kr(1.0, 2.0, "a"), kr(2.5, 4.0, "b"), kr(10.0, 11.0, "c")],
        dubbed=[en(1.0, 2.0, "a"), en(2.5, 4.0, "b"), en(10.0, 11.0, "c")],
    )
    pairs = assign_scenes(pairs, gap_threshold=3.0)
    assert pairs[0].scene_id == pairs[1].scene_id == "scene_1"
    assert pairs[2].scene_id == "scene_2"
    scenes = group_by_scene(pairs)
    assert len(scenes["scene_1"]) == 2
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 구현**

`backend/app/core/alignment.py`:

```python
from typing import Dict, List
from app.schemas import SegmentText, AlignedPair


def _overlap(a: SegmentText, b: SegmentText) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def align(korean: List[SegmentText], dubbed: List[SegmentText]) -> List[AlignedPair]:
    pairs: List[AlignedPair] = []
    used_dubbed = set()
    for i, kr in enumerate(korean):
        best_j, best_ov = None, 0.0
        for j, en in enumerate(dubbed):
            if j in used_dubbed:
                continue
            ov = _overlap(kr, en)
            if ov > best_ov:
                best_j, best_ov = j, ov
        if best_j is not None:
            en = dubbed[best_j]
            used_dubbed.add(best_j)
            union = max(kr.end, en.end) - min(kr.start, en.start)
            conf = round(best_ov / union, 3) if union > 0 else 0.0
            pairs.append(AlignedPair(id=f"pair_{i+1}", korean=kr, dubbed=en,
                                     alignment_confidence=conf))
        else:
            pairs.append(AlignedPair(id=f"pair_{i+1}", korean=kr, dubbed=None,
                                     alignment_confidence=0.0))
    for j, en in enumerate(dubbed):
        if j not in used_dubbed:
            pairs.append(AlignedPair(id=f"extra_{j+1}", korean=None, dubbed=en,
                                     alignment_confidence=0.0))
    pairs.sort(key=lambda p: (p.korean or p.dubbed).start)
    return pairs


def assign_scenes(pairs: List[AlignedPair], gap_threshold: float = 3.0) -> List[AlignedPair]:
    scene_num = 1
    prev_end = None
    for p in pairs:
        anchor = p.korean or p.dubbed
        if prev_end is not None and anchor.start - prev_end > gap_threshold:
            scene_num += 1
        p.scene_id = f"scene_{scene_num}"
        prev_end = max(prev_end or 0.0, anchor.end)
    return pairs


def group_by_scene(pairs: List[AlignedPair]) -> Dict[str, List[AlignedPair]]:
    scenes: Dict[str, List[AlignedPair]] = {}
    for p in pairs:
        scenes.setdefault(p.scene_id, []).append(p)
    return scenes
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/alignment.py backend/tests/test_alignment.py
git commit -m "feat: 한↔영 타임코드 정렬 + 간격 기반 씬 배정"
```

---

### Task 6: 결정론적 룰 체크 — 텍스트 계열 (누락/발화속도/싱크/정렬 신뢰도)

**Files:**
- Create: `backend/app/core/rule_checks.py`
- Test: `backend/tests/test_rule_checks.py`

**Interfaces:**
- Consumes: `AlignedPair`, `QCFinding`
- Produces:
  - `check_missing(pairs) -> list[QCFinding]` — dubbed 없음/빈 텍스트 → high, axis 언어 적합성, issue_type "번역 누락"
  - `check_pacing(pairs, max_words_per_sec: float = 3.8) -> list[QCFinding]` — axis 싱크 정확도, issue_type "발화속도 초과"
  - `check_sync_overflow(pairs, tolerance: float = 0.5) -> list[QCFinding]` — dubbed 구간이 korean 구간을 tolerance 초 초과 이탈 → axis 싱크 정확도, issue_type "싱크 오버플로"
  - `check_low_alignment(pairs, min_confidence: float = 0.3) -> list[QCFinding]` — axis 싱크 정확도, issue_type "정렬 신뢰도 저하"
  - `run_text_checks(pairs) -> list[QCFinding]` — 위 4개 순차 실행 후 합침. 모든 finding의 `source="rule"`, `category="localization"`(싱크 계열은 `category="voice"`), id는 `rule_{종류}_{pair.id}`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_rule_checks.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_rule_checks.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 구현**

`backend/app/core/rule_checks.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_rule_checks.py -v`
Expected: 6 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/rule_checks.py backend/tests/test_rule_checks.py
git commit -m "feat: 결정론적 룰 체크 (누락/발화속도/싱크/정렬 신뢰도)"
```

---

### Task 7: 룰 체크 — 오디오 계열 (음질) + SRT↔음성 일치 검증

**Files:**
- Modify: `backend/app/core/rule_checks.py` (함수 추가)
- Test: `backend/tests/test_audio_checks.py`

**Interfaces:**
- Consumes: Task 6의 `_finding`, `ModelProvider`
- Produces:
  - `read_wav_mono(path: str) -> tuple[list[int], int]` — 16-bit mono WAV → (샘플, 샘플레이트)
  - `check_audio_quality(wav_path: str, pairs: list[AlignedPair]) -> list[QCFinding]` — 클리핑(|s|≥32700 비율>0.1%), 세그먼트 내 드롭아웃(RMS<100), SNR(상위20% RMS vs 하위10% RMS < 15dB) → axis 음질, category voice
  - `async check_srt_audio_match(pairs, stem_wav_path, provider, extract_clip_fn, sample_every: int = 10) -> list[QCFinding]` — N개마다 1개 샘플링, 클립 STT 후 토큰 자카드 유사도 < 0.4 → issue_type "자막-음성 불일치", axis 언어 적합성
  - `extract_clip(src: str, start: float, end: float) -> str` — ffmpeg로 구간 WAV 추출, 출력 경로 반환

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_audio_checks.py`:

```python
import math
import struct
import wave
from app.core.rule_checks import (
    read_wav_mono, check_audio_quality, check_srt_audio_match, _token_similarity,
)
from app.providers.base import get_provider
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


async def test_srt_audio_match_flags_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    # MockProvider.transcribe는 한국어 고정 대사를 반환 → 영어 SRT와 불일치
    pairs = [pair_at(0.0, 2.0, en="totally unrelated english line", pid="p1")]

    def fake_clip(src, start, end):
        return src  # 실제 ffmpeg 호출 없이 원본 경로 반환

    findings = await check_srt_audio_match(
        pairs, "/tmp/stem.wav", provider, extract_clip_fn=fake_clip, sample_every=1,
    )
    assert len(findings) == 1
    assert findings[0].issue_type == "자막-음성 불일치"
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_audio_checks.py -v`
Expected: FAIL — ImportError (`read_wav_mono` 등 미정의)

- [ ] **Step 3: 구현** — `backend/app/core/rule_checks.py` 끝에 추가:

```python
import math
import os
import struct
import subprocess
import tempfile
import wave
from app.providers.base import ModelProvider


def read_wav_mono(path: str):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "16-bit mono WAV 필요"
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = list(struct.unpack(f"{len(raw) // 2}h", raw))
    return samples, rate


def _rms(chunk) -> float:
    if not chunk:
        return 0.0
    return math.sqrt(sum(s * s for s in chunk) / len(chunk))


def check_audio_quality(wav_path: str, pairs: List[AlignedPair]) -> List[QCFinding]:
    samples, rate = read_wav_mono(wav_path)
    findings: List[QCFinding] = []
    if not samples:
        return findings

    # 1) 클리핑: 최대치 근접 샘플 비율
    clipped = sum(1 for s in samples if abs(s) >= 32700)
    if clipped / len(samples) > 0.001:
        anchor = pairs[0] if pairs else None
        if anchor:
            findings.append(_finding(
                "clipping", anchor, "high", "클리핑", "음질",
                f"오디오 샘플의 {clipped / len(samples) * 100:.2f}%가 클리핑되었습니다. "
                "왜곡된 구간의 재녹음/마스터링 확인이 필요합니다.",
                "Re-master or re-record the clipped sections.", category="voice",
            ))

    # 2) 세그먼트 내 드롭아웃: 대사 구간인데 0.3초 이상 RMS<100 연속
    frame = rate // 10  # 100ms
    for p in pairs:
        seg = p.dubbed or p.korean
        if seg is None:
            continue
        lo, hi = int(seg.start * rate), min(int(seg.end * rate), len(samples))
        silent_run = 0
        found = False
        for i in range(lo, hi, frame):
            if _rms(samples[i:i + frame]) < 100:
                silent_run += 1
                if silent_run >= 3 and not found:  # 300ms 이상
                    findings.append(_finding(
                        "dropout", p, "high", "드롭아웃", "음질",
                        "대사 구간 안에 0.3초 이상의 완전 무음이 있습니다. "
                        "오디오 누락 여부를 확인하세요.",
                        "Check for missing audio in this segment.", category="voice",
                    ))
                    found = True
            else:
                silent_run = 0

    # 3) SNR: 상위 20% 프레임 RMS 대비 하위 10% 프레임 RMS
    frame_rms = sorted(_rms(samples[i:i + frame]) for i in range(0, len(samples), frame))
    if len(frame_rms) >= 10:
        noise = frame_rms[max(0, int(len(frame_rms) * 0.1) - 1)] or 1.0
        speech = frame_rms[int(len(frame_rms) * 0.8)]
        snr_db = 20 * math.log10(speech / noise) if noise > 0 and speech > 0 else 99
        if snr_db < 15:
            anchor = pairs[0] if pairs else None
            if anchor:
                findings.append(_finding(
                    "snr", anchor, "medium", "잡음", "음질",
                    f"추정 SNR이 {snr_db:.0f}dB로 낮습니다. 배경 잡음 확인이 필요합니다.",
                    "Reduce background noise in the dialogue stem.", category="voice",
                ))
    return findings


def _token_similarity(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_clip(src: str, start: float, end: float) -> str:
    out = os.path.join(tempfile.gettempdir(), f"qc_clip_{start:.1f}_{end:.1f}.wav")
    subprocess.run(
        ["ffmpeg", "-i", src, "-ss", str(start), "-to", str(end),
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return out


async def check_srt_audio_match(pairs: List[AlignedPair], stem_wav_path: str,
                                provider: ModelProvider, extract_clip_fn=extract_clip,
                                sample_every: int = 10) -> List[QCFinding]:
    findings = []
    targets = [p for p in pairs if p.dubbed and p.dubbed.text.strip()][::sample_every]
    for p in targets:
        clip = extract_clip_fn(stem_wav_path, p.dubbed.start, p.dubbed.end)
        heard = await provider.transcribe(clip, lang="en")
        heard_text = " ".join(s.text for s in heard)
        if _token_similarity(p.dubbed.text, heard_text) < 0.4:
            findings.append(_finding(
                "srtmatch", p, "medium", "자막-음성 불일치", "언어 적합성",
                f"SRT 자막과 실제 더빙 음성이 다르게 들립니다. "
                f"(음성 인식 결과: \"{heard_text[:80]}\") 누락/애드리브/다른 테이크 여부를 확인하세요.",
                "Verify the recorded line against the final script.",
            ))
    return findings
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_audio_checks.py -v`
Expected: 6 passed. 기존 테스트 회귀 확인: `venv/bin/python -m pytest -q` → all passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/rule_checks.py backend/tests/test_audio_checks.py
git commit -m "feat: 음질 체크(클리핑/드롭아웃/SNR) + SRT-음성 일치 검증"
```

---

### Task 8: 지식베이스 (YAML + 로더)

**Files:**
- Create: `backend/app/knowledge/honorifics.yaml`
- Create: `backend/app/knowledge/idioms.yaml`
- Create: `backend/app/knowledge/loader.py`
- Create: `backend/app/knowledge/__init__.py`
- Test: `backend/tests/test_knowledge.py`

**Interfaces:**
- Produces: `load_knowledge(dir_path: str | None = None) -> str` — YAML 항목들을 프롬프트 주입용 한국어 불릿 텍스트로 변환. 인자 생략 시 `backend/app/knowledge/` 사용.

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_knowledge.py`:

```python
from app.knowledge.loader import load_knowledge


def test_load_default_knowledge_contains_core_rules():
    text = load_knowledge()
    assert "형" in text
    assert "눈치" in text
    assert "어이가 없네" in text


def test_load_knowledge_from_custom_dir(tmp_path):
    (tmp_path / "custom.yaml").write_text(
        "rules:\n  - term: 테스트어\n    rule: 테스트 규칙\n    bad: bad ex\n    good: good ex\n",
        encoding="utf-8",
    )
    text = load_knowledge(str(tmp_path))
    assert "테스트어" in text
    assert "good ex" in text
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_knowledge.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 구현**

`backend/app/knowledge/__init__.py`: 빈 파일.

`backend/app/knowledge/honorifics.yaml`:

```yaml
# 호칭 처리 규칙 — 검수자가 직접 추가/수정 가능
rules:
  - term: 형 / 오빠 / 누나 / 언니
    rule: 친족이 아닌 친근한 호칭은 brother/sister로 직역 금지. 이름, man, hey 등으로 의역.
    bad: "Brother, did you eat?"
    good: "Hey man, you eaten?"
  - term: 선배 / 후배
    rule: senior/junior 직역 금지. 이름 호칭 또는 생략. 위계가 서사에 중요하면 대사 톤으로 표현.
    bad: "Senior, please help me."
    good: "Ji-hoon, please help me."
  - term: 부장님 / 과장님 / 사장님
    rule: 직급 직역(Manager Kim) 대신 Mr./Ms. + 성 또는 직함 생략. 격식은 문체로 전달.
    bad: "Department Head Kim, hello."
    good: "Mr. Kim, hello."
  - term: 존댓말 → 반말 전환 (또는 그 반대)
    rule: 한국어의 말단 높임 전환은 관계 변화의 연출. 영어에서는 호칭 변화, 격식 어휘, 문장 길이로 등가 표현 필요. 전환이 사라졌으면 지적할 것.
    bad: (전환 전후가 동일한 캐주얼 영어)
    good: (전환 후 격식체/거리감 있는 표현으로 변화)
```

`backend/app/knowledge/idioms.yaml`:

```yaml
# 관용구 사전 — 검수자가 직접 추가/수정 가능
rules:
  - term: 눈치 (보다/채다/없다)
    rule: eye/look 직역 금지. read the room, walk on eggshells, take a hint 등 상황별 의역.
    bad: "Stop looking at my eyes."
    good: "Stop walking on eggshells."
  - term: 어이가 없네
    rule: 신체 부위 오역(kidney 등) 치명 오류. ridiculous, unbelievable, speechless 계열로.
    bad: "I have no kidney."
    good: "This is ridiculous."
  - term: 밥 먹었어? / 밥은 먹고 다니냐
    rule: 식사 여부가 아닌 안부 인사면 How are you / You doing okay 계열. 실제 식사 제안이면 직역 유지.
    bad: "Did you eat rice?"
    good: "Have you eaten? / You doing okay?"
  - term: 수고하셨습니다
    rule: You suffered 직역 금지. Good work / Thanks for today 등.
    bad: "You suffered a lot."
    good: "Great work today."
```

`backend/app/knowledge/loader.py`:

```python
import os
from pathlib import Path
import yaml

_DEFAULT_DIR = Path(__file__).parent


def load_knowledge(dir_path: str | None = None) -> str:
    base = Path(dir_path) if dir_path else _DEFAULT_DIR
    lines = []
    for yml in sorted(base.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        for rule in data.get("rules", []):
            lines.append(
                f"- {rule.get('term', '')}: {rule.get('rule', '')}"
                f" (나쁜 예: {rule.get('bad', '-')} / 좋은 예: {rule.get('good', '-')})"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_knowledge.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/knowledge backend/tests/test_knowledge.py
git commit -m "feat: 호칭/관용구 지식베이스 (YAML) + 프롬프트 로더"
```

---

### Task 9: 페르소나 패널 + 병합기 (judge_panel)

**Files:**
- Create: `backend/app/core/judge_panel.py`
- Test: `backend/tests/test_judge_panel.py`

**Interfaces:**
- Consumes: `ModelProvider`, `Persona`, `AlignedPair`, `QCFinding`, Task 7의 `extract_clip`
- Produces:
  - `PERSONAS: list[Persona]` — key가 정확히 `"culture"`, `"native"`, `"director"`인 3개 (director만 `uses_audio=True`)
  - `merge_findings(findings: list[QCFinding]) -> list[QCFinding]` — 같은 `segment_id` 그룹: `agreement`=서로 다른 persona 수, severity=최고치, `alternatives`={페르소나 이름: recommendation}, description은 최고 severity finding 기준
  - `async run_panel(scenes: dict[str, list[AlignedPair]], knowledge: str, provider, stem_wav_path: str | None = None, on_progress=None) -> list[QCFinding]` — 씬별 × 페르소나별 judge 호출(director는 씬 오디오 클립 첨부), 청크 실패 시 1회 재시도 후 해당 씬 건너뜀, `on_progress(done_scenes, total_scenes)` 콜백

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_judge_panel.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_judge_panel.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 구현**

`backend/app/core/judge_panel.py`:

```python
from typing import Callable, Dict, List, Optional
from app.providers.base import ModelProvider, Persona
from app.schemas import AlignedPair, QCFinding

PERSONAS: List[Persona] = [
    Persona(
        key="culture", name="한국 문화·언어 전문가",
        axes=["언어 적합성"],
        instruction=(
            "당신의 임무는 영어 문장이 아무리 자연스러워도 한국어 원문의 뉘앙스가 "
            "훼손되었는지 잡아내는 것입니다. 확인할 것: (1) 호칭(형/누나/선배/부장님)의 "
            "관계 정보가 보존되는가 (2) 존댓말↔반말 전환 같은 화계 변화가 영어에서 "
            "등가로 표현되는가 (3) 관용구가 문맥에 맞게 의역되었는가 "
            "(4) 반어법/눈치 대사가 곧이곧대로 번역되지 않았는가. "
            "영어 자체의 유창성은 평가하지 마십시오 — 그건 다른 검수자의 몫입니다."
        ),
    ),
    Persona(
        key="native", name="영어 원어민 시청자",
        axes=["자연스러움", "언어 적합성"],
        instruction=(
            "당신은 한국어를 전혀 모르는 미국인 시청자입니다. korean 필드는 무시하고 "
            "english_dub만 읽으십시오. 확인할 것: (1) 원어민이 실제로 쓰는 표현인가, "
            "번역투인가 (2) 구어체 대사로서 리듬이 자연스러운가 (3) 어색하거나 "
            "우스꽝스럽게 들리는 문장이 있는가. 의미의 정확성은 평가하지 마십시오."
        ),
    ),
    Persona(
        key="director", name="더빙 연출가",
        axes=["감정 표현", "자연스러움"], uses_audio=True,
        instruction=(
            "당신은 더빙 연출가입니다. 씬의 정서와 캐릭터를 기준으로 평가하십시오. "
            "확인할 것: (1) 대사 톤이 씬 맥락(대립/화해/긴장)과 맞는가 "
            "(2) 캐릭터 성격과 말투가 일관되는가 (3) 오디오가 제공되면 성우의 "
            "감정 표현·억양이 대사 내용과 어울리는가 (4) 대사 호흡이 장면 리듬에 맞는가."
        ),
    ),
]


def _severity_rank(s: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(s, 0)


def merge_findings(findings: List[QCFinding]) -> List[QCFinding]:
    by_segment: Dict[str, List[QCFinding]] = {}
    for f in findings:
        by_segment.setdefault(f.segment_id, []).append(f)
    merged = []
    persona_names = {p.key: p.name for p in PERSONAS}
    for seg_id, group in by_segment.items():
        primary = max(group, key=lambda f: (_severity_rank(f.severity), f.confidence))
        personas = {f.source for f in group if f.source.startswith("persona:")}
        alternatives = {}
        for f in group:
            if f.source.startswith("persona:") and f.recommendation:
                key = f.source.split(":", 1)[1]
                alternatives[persona_names.get(key, key)] = f.recommendation
        primary = primary.model_copy(update={
            "agreement": max(len(personas), 1),
            "alternatives": alternatives,
        })
        merged.append(primary)
    merged.sort(key=lambda f: f.start_time)
    return merged


async def run_panel(scenes: Dict[str, List[AlignedPair]], knowledge: str,
                    provider: ModelProvider, stem_wav_path: Optional[str] = None,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> List[QCFinding]:
    from app.core.rule_checks import extract_clip

    all_findings: List[QCFinding] = []
    scene_ids = sorted(scenes.keys(), key=lambda s: int(s.split("_")[1]))
    for done, scene_id in enumerate(scene_ids, start=1):
        pairs = scenes[scene_id]
        clip_path = None
        if stem_wav_path:
            anchors = [(p.dubbed or p.korean) for p in pairs if (p.dubbed or p.korean)]
            if anchors:
                clip_path = extract_clip(stem_wav_path, anchors[0].start, anchors[-1].end)
        for persona in PERSONAS:
            audio = clip_path if persona.uses_audio else None
            for attempt in (1, 2):
                try:
                    all_findings.extend(
                        await provider.judge(pairs, persona, knowledge, audio_clip_path=audio)
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[패널] {scene_id}/{persona.key} 분석 실패 (2회 시도): {e}")
        if on_progress:
            on_progress(done, len(scene_ids))
    return merge_findings(all_findings)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_judge_panel.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/judge_panel.py backend/tests/test_judge_panel.py
git commit -m "feat: 3-페르소나 Judge 패널 + agreement 병합기"
```

---

### Task 10: 판정 엔진 (verdict + 설정 파일)

**Files:**
- Create: `backend/app/qc_config.yaml`
- Create: `backend/app/core/verdict.py`
- Test: `backend/tests/test_verdict.py`

**Interfaces:**
- Consumes: `QCFinding`, `AxisScore`, `Verdict`, `AXES`
- Produces:
  - `load_config(path: str | None = None) -> dict`
  - `compute_axis_scores(findings: list[QCFinding], n_pairs: int, config: dict) -> list[AxisScore]` — 축별 감점 합계를 100세그먼트당 비율로 정규화 후 MOS 매핑. **5축 모두 항상 반환** (finding 없는 축은 MOS 5)
  - `decide(axis_scores: list[AxisScore], findings: list[QCFinding], config: dict) -> Verdict` — 전 축 ≥4 → pass / 최저 축 3 → conditional / 어느 축 ≤2 또는 high finding 존재 → fail (fail 우선)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_verdict.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_verdict.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 구현**

`backend/app/qc_config.yaml`:

```yaml
# 판정 임계값 설정 — 운영하며 조정
deduction:            # finding 1건당 감점
  high: 15
  medium: 8
  low: 3
mos_thresholds:       # 100세그먼트당 감점률 상한 → MOS
  5: 5
  4: 15
  3: 35
  2: 60
  # 60 초과 → 1
verdict:
  pass_min_mos: 4       # 전 축 이 값 이상이면 통과
  conditional_min_mos: 3  # 최저 축이 이 값이면 조건부 통과
```

`backend/app/core/verdict.py`:

```python
from pathlib import Path
from typing import List, Optional
import yaml
from app.schemas import QCFinding, AxisScore, Verdict, AXES

_DEFAULT_CONFIG = Path(__file__).parent.parent / "qc_config.yaml"


def load_config(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _DEFAULT_CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def compute_axis_scores(findings: List[QCFinding], n_pairs: int, config: dict) -> List[AxisScore]:
    n_pairs = max(n_pairs, 1)
    deduction_w = config["deduction"]
    thresholds = config["mos_thresholds"]
    scores = []
    for axis in AXES:
        total = sum(deduction_w.get(f.severity, 0) for f in findings if f.axis == axis)
        rate = round(total / n_pairs * 100, 1)
        mos = 1
        for level in (5, 4, 3, 2):
            if rate <= thresholds[level]:
                mos = level
                break
        scores.append(AxisScore(axis=axis, mos=mos, deduction_rate=rate))
    return scores


def decide(axis_scores: List[AxisScore], findings: List[QCFinding], config: dict) -> Verdict:
    reasons = []
    high_findings = [f for f in findings if f.severity == "high"]
    min_mos = min(s.mos for s in axis_scores)
    pass_min = config["verdict"]["pass_min_mos"]
    cond_min = config["verdict"]["conditional_min_mos"]

    if high_findings:
        reasons.append(f"치명(high) 지적 {len(high_findings)}건 — 심각도 무관 즉시 반려 대상입니다.")
    for s in axis_scores:
        if s.mos < cond_min:
            reasons.append(f"{s.axis} MOS {s.mos} (감점률 {s.deduction_rate})")

    if high_findings or min_mos < cond_min:
        status = "fail"
    elif min_mos < pass_min:
        status = "conditional"
        reasons.append(f"최저 축 MOS {min_mos} — 수정 권고 후 통과 가능합니다.")
    else:
        status = "pass"
    return Verdict(status=status, axis_scores=axis_scores, reasons=reasons)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_verdict.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/qc_config.yaml backend/app/core/verdict.py backend/tests/test_verdict.py
git commit -m "feat: 5축 MOS 판정 엔진 + 임계값 설정 파일"
```

---

### Task 11: 피드백 저장소 (JSONL)

**Files:**
- Create: `backend/app/feedback/__init__.py`
- Create: `backend/app/feedback/store.py`
- Test: `backend/tests/test_feedback_store.py`

**Interfaces:**
- Consumes: `FeedbackEntry`
- Produces:
  - `FeedbackStore(path: str)` — `.record(entry: FeedbackEntry) -> None` (timestamp 자동 기입, append), `.all() -> list[dict]`
  - 기본 경로: `backend/data/feedback.jsonl` (main.py에서 지정)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_feedback_store.py`:

```python
from app.feedback.store import FeedbackStore
from app.schemas import FeedbackEntry


def entry(action="approved", final=""):
    return FeedbackEntry(
        movie="테스트영화", segment_id="pair_1", korean="어이가 없네",
        dubbed="I have no kidney", finding_id="f1",
        reviewer_action=action, final_text=final,
    )


def test_record_appends_jsonl_with_timestamp(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.jsonl"))
    store.record(entry())
    store.record(entry(action="modified", final="This is ridiculous."))
    rows = store.all()
    assert len(rows) == 2
    assert rows[0]["reviewer_action"] == "approved"
    assert rows[0]["timestamp"] != ""
    assert rows[1]["final_text"] == "This is ridiculous."


def test_store_creates_parent_dir(tmp_path):
    store = FeedbackStore(str(tmp_path / "nested" / "fb.jsonl"))
    store.record(entry())
    assert len(store.all()) == 1


def test_all_returns_empty_for_missing_file(tmp_path):
    store = FeedbackStore(str(tmp_path / "none.jsonl"))
    assert store.all() == []
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_feedback_store.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 구현**

`backend/app/feedback/__init__.py`: 빈 파일.

`backend/app/feedback/store.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from app.schemas import FeedbackEntry


class FeedbackStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: FeedbackEntry) -> None:
        data = entry.model_dump()
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def all(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_feedback_store.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/feedback backend/tests/test_feedback_store.py
git commit -m "feat: 검수 피드백 JSONL 저장소 (학습 데이터 축적 입구)"
```

---

### Task 12: 파이프라인 오케스트레이션 재작성

**Files:**
- Modify: `backend/app/core/pipeline.py` (전면 재작성)
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 2~10의 전부 — `get_provider`, `load_text_source`, `align`, `assign_scenes`, `group_by_scene`, `run_text_checks`, `check_audio_quality`, `check_srt_audio_match`, `load_knowledge`, `run_panel`, `compute_axis_scores`, `decide`, `load_config`
- Produces:
  - `QCPipeline(provider: ModelProvider | None = None)` — None이면 실행 시 `get_provider()` 호출
  - `async QCPipeline.run(job: QCJobInput, on_progress: Callable[[str, int, int], None] | None = None) -> QCResult` — 진행 콜백 `(stage, done, total)`, stage ∈ {"ingest", "align", "rules", "panel", "verdict"}
- 참고: 기존 `QCPipeline.run(QCRequest)`은 삭제된다. `context.py`/`voice_qc.py`는 파일 유지하되 새 파이프라인에서 호출하지 않음 (시각 맥락·음색 검사는 실모델 연동 시 재활성화 — 스펙 §10 범위 밖 참조).

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_pipeline.py`:

```python
import math
import struct
import wave
import pytest
from app.core.pipeline import QCPipeline
from app.providers.base import get_provider
from app.schemas import QCJobInput

EN_SRT = """1
00:00:01,000 --> 00:00:03,000
Hey brother, did you eat rice?

2
00:00:04,000 --> 00:00:06,000
I have no kidney.
"""

KR_SRT = """1
00:00:01,000 --> 00:00:03,000
형, 밥 먹었어?

2
00:00:04,000 --> 00:00:06,000
어이가 없네.
"""


@pytest.fixture
def job_files(tmp_path):
    en = tmp_path / "en.srt"
    en.write_text(EN_SRT, encoding="utf-8")
    kr = tmp_path / "kr.srt"
    kr.write_text(KR_SRT, encoding="utf-8")
    stem = tmp_path / "stem.wav"
    rate, samples = 16000, []
    for i in range(rate * 7):
        samples.append(int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
    with wave.open(str(stem), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack(f"{len(samples)}h", *samples))
    return str(en), str(kr), str(stem)


async def test_pipeline_end_to_end_with_srt_both(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    en, kr, stem = job_files
    stages = []
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(
        QCJobInput(movie_title="테스트", en_srt_path=en, kr_srt_path=kr, stem_audio_path=stem),
        on_progress=lambda stage, d, t: stages.append(stage),
    )
    assert result.verdict.status == "fail"  # kidney → high → 즉시 반려
    assert len(result.pairs) == 2
    seg_findings = [f for f in result.findings if f.source.startswith("persona:")]
    assert any("kidney" in f.current_translation for f in seg_findings)
    assert {"ingest", "align", "rules", "panel", "verdict"} <= set(stages)


async def test_pipeline_without_stem_skips_audio_checks(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    en, kr, _ = job_files
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(QCJobInput(en_srt_path=en, kr_srt_path=kr))
    assert all(f.issue_type not in ("클리핑", "드롭아웃", "잡음") for f in result.findings)
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — 기존 QCPipeline 시그니처와 불일치 (ImportError 또는 TypeError)

- [ ] **Step 3: pipeline.py 전면 재작성**

`backend/app/core/pipeline.py`:

```python
from typing import Callable, Optional
from app.schemas import QCJobInput, QCResult
from app.providers.base import ModelProvider, get_provider
from app.core.ingest import load_text_source
from app.core.alignment import align, assign_scenes, group_by_scene
from app.core.rule_checks import run_text_checks, check_audio_quality, check_srt_audio_match
from app.core.judge_panel import run_panel
from app.core.verdict import load_config, compute_axis_scores, decide
from app.knowledge.loader import load_knowledge

ProgressFn = Callable[[str, int, int], None]


class QCPipeline:
    def __init__(self, provider: Optional[ModelProvider] = None):
        self.provider = provider

    async def run(self, job: QCJobInput, on_progress: Optional[ProgressFn] = None) -> QCResult:
        provider = self.provider or get_provider()
        notify = on_progress or (lambda stage, d, t: None)

        # ① 텍스트 수집 (SRT 우선, STT 폴백)
        notify("ingest", 0, 2)
        korean = await load_text_source("ko", job.kr_srt_path, job.kr_audio_path, provider)
        notify("ingest", 1, 2)
        dubbed = await load_text_source("en", job.en_srt_path, None, provider)
        notify("ingest", 2, 2)

        # ② 정렬 + 씬 배정
        notify("align", 0, 1)
        pairs = assign_scenes(align(korean, dubbed))
        notify("align", 1, 1)

        # ③ 결정론적 룰 체크
        notify("rules", 0, 1)
        findings = run_text_checks(pairs)
        if job.stem_audio_path:
            findings += check_audio_quality(job.stem_audio_path, pairs)
            findings += await check_srt_audio_match(pairs, job.stem_audio_path, provider)
        notify("rules", 1, 1)

        # ④ 페르소나 패널
        scenes = group_by_scene(pairs)
        panel_findings = await run_panel(
            scenes, load_knowledge(), provider,
            stem_wav_path=job.stem_audio_path,
            on_progress=lambda d, t: notify("panel", d, t),
        )
        findings += panel_findings

        # ⑤ 판정
        notify("verdict", 0, 1)
        config = load_config()
        axis_scores = compute_axis_scores(findings, n_pairs=len(pairs), config=config)
        verdict = decide(axis_scores, findings, config)
        notify("verdict", 1, 1)

        return QCResult(verdict=verdict, findings=findings, pairs=pairs)
```

- [ ] **Step 4: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 passed (test_pipeline 2건 포함)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: QC 파이프라인 재작성 (ingest→정렬→룰체크→패널→판정)"
```

---

### Task 13: API 개편 (main.py)

**Files:**
- Modify: `backend/app/main.py` (대폭 개편)
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `QCPipeline`, `FeedbackStore`, `get_provider`, `ProviderNotConfiguredError`, 스키마 전부
- Produces (프론트엔드가 사용):
  - `POST /api/qc/upload-media?role={original|dubbed|stem|srt_en|srt_kr}` — srt 계열은 파일 저장만, 미디어 계열은 기존 ffmpeg 추출+waveform 유지. 응답: `{success, role, filename, media_path, audio_path?, waveform?}`
  - `POST /api/qc/run` body=`QCJobInput` → `{job_id}` (202). 키 없으면 503 `{detail}`
  - `GET /api/qc/jobs/{job_id}` → `{status: "queued"|"running"|"done"|"error", progress: {stage, done, total}, result?: QCResult, error?}`
  - `POST /api/qc/feedback` body=`FeedbackEntry` → `{ok: true}`
  - `POST /api/qc/jobs/{job_id}/reverdict` body=`{"excluded_finding_ids": ["f1", ...]}` → 반려(오탐) finding 제외 후 재계산한 `Verdict` (스펙 §7 규칙 1: AI 판정은 가판정, 검수자 확정 후 재판정)
  - `GET /api/qc/export/{job_id}` → 확정 finding CSV (UTF-8 BOM, 엑셀 호환)
  - **삭제:** `/api/qc/translate`, `/api/qc/mock-data`, `/api/qc/transcribe`, `/api/qc/process`, `/api/qc/upload-video` — STT 폴백/mock 반환 로직 전부 제거

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_api.py`:

```python
import time
import pytest
from fastapi.testclient import TestClient

EN_SRT = """1
00:00:01,000 --> 00:00:03,000
I have no kidney.
"""

KR_SRT = """1
00:00:01,000 --> 00:00:03,000
어이가 없네.
"""


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setenv("QC_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    from app.main import app
    return TestClient(app)


def _run_job(client, tmp_path):
    en = tmp_path / "en.srt"; en.write_text(EN_SRT, encoding="utf-8")
    kr = tmp_path / "kr.srt"; kr.write_text(KR_SRT, encoding="utf-8")
    res = client.post("/api/qc/run", json={
        "movie_title": "t", "en_srt_path": str(en), "kr_srt_path": str(kr),
    })
    assert res.status_code == 202
    job_id = res.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/qc/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job_id, job
        time.sleep(0.1)
    pytest.fail("job did not finish")


def test_run_and_poll_job(client, tmp_path):
    job_id, job = _run_job(client, tmp_path)
    assert job["status"] == "done"
    assert job["result"]["verdict"]["status"] == "fail"
    assert len(job["result"]["pairs"]) == 1


def test_run_without_provider_returns_503(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("QC_PROVIDER", "gemini")
    from app.main import app
    client = TestClient(app)
    en = tmp_path / "en.srt"; en.write_text(EN_SRT, encoding="utf-8")
    res = client.post("/api/qc/run", json={"en_srt_path": str(en)})
    assert res.status_code == 503
    assert "GEMINI_API_KEY" in res.json()["detail"]


def test_feedback_recorded(client):
    res = client.post("/api/qc/feedback", json={
        "movie": "t", "segment_id": "pair_1", "korean": "어이가 없네",
        "dubbed": "I have no kidney", "finding_id": "f1",
        "reviewer_action": "modified", "final_text": "This is ridiculous.",
    })
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_export_csv(client, tmp_path):
    job_id, _ = _run_job(client, tmp_path)
    res = client.get(f"/api/qc/export/{job_id}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "kidney" in res.text


def test_reverdict_excluding_high_finding_lifts_fail(client, tmp_path):
    job_id, job = _run_job(client, tmp_path)
    assert job["result"]["verdict"]["status"] == "fail"
    high_ids = [f["id"] for f in job["result"]["findings"] if f["severity"] == "high"]
    res = client.post(f"/api/qc/jobs/{job_id}/reverdict",
                      json={"excluded_finding_ids": high_ids})
    assert res.status_code == 200
    assert res.json()["status"] != "fail"  # high 오탐 제외 → 반려 해제


def test_removed_endpoints_are_gone(client):
    assert client.get("/api/qc/mock-data").status_code == 404
    assert client.post("/api/qc/translate", json={"segments": []}).status_code == 404
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL (신규 엔드포인트 404, 구 엔드포인트 존재)

- [ ] **Step 3: main.py 개편**

`backend/app/main.py`를 아래로 교체 (.env 로더, CORS, upload-media의 ffmpeg/waveform 로직은 기존 코드 유지):

```python
import os
import csv
import io
import uuid
import asyncio

# .env 로더 — 기존 코드 그대로 유지 (main.py 상단 1~15행)
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip().strip("\"'")

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import QCJobInput, FeedbackEntry
from app.core.pipeline import QCPipeline
from app.providers.base import get_provider, ProviderNotConfiguredError
from app.feedback.store import FeedbackStore
import shutil
import tempfile
import subprocess
import struct

app = FastAPI(title="AI Dubbing QC API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

JOBS: dict = {}
VALID_ROLES = {"original", "dubbed", "stem", "srt_en", "srt_kr"}


def _feedback_store() -> FeedbackStore:
    path = os.getenv("QC_FEEDBACK_PATH",
                     os.path.join(os.path.dirname(__file__), "..", "data", "feedback.jsonl"))
    return FeedbackStore(path)


@app.get("/")
def read_root():
    return {"message": "AI Dubbing QC Backend API is running."}


@app.post("/api/qc/upload-media")
async def upload_media(file: UploadFile = File(...), role: str = "dubbed"):
    if role not in VALID_ROLES:
        raise HTTPException(400, f"role은 {sorted(VALID_ROLES)} 중 하나여야 합니다.")
    temp_dir = tempfile.gettempdir()
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    media_path = os.path.join(temp_dir, f"{role}_{safe_filename}")
    with open(media_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if role.startswith("srt_"):
        return {"success": True, "role": role, "filename": file.filename,
                "media_path": media_path}

    # 미디어(원본/더빙본/스템): 16kHz mono WAV 추출 + waveform peaks — 기존 로직 그대로
    audio_filename = f"{role}_{os.path.splitext(safe_filename)[0]}.wav"
    audio_path = os.path.join(temp_dir, audio_filename)
    raw_audio_path = os.path.join(temp_dir, f"{role}_{os.path.splitext(safe_filename)[0]}_peaks.raw")
    try:
        subprocess.run(["ffmpeg", "-i", media_path, "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1", "-y", audio_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # NOTE: 저레이트 리샘플 금지 — 안티앨리어싱 필터가 음성 에너지를 제거함.
        # 시각화용 다운샘플은 아래 max-per-bin으로 수행.
        subprocess.run(["ffmpeg", "-i", media_path, "-f", "s16le", "-ac", "1",
                        "-ar", "16000", "-y", raw_audio_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        peaks = []
        if os.path.exists(raw_audio_path):
            with open(raw_audio_path, "rb") as f:
                raw_data = f.read()
            num_samples = len(raw_data) // 2
            if num_samples > 0:
                samples = struct.unpack(f"{num_samples}h", raw_data)
                bin_size = max(1, num_samples // 600)
                for i in range(0, num_samples, bin_size):
                    chunk = samples[i:i + bin_size]
                    if chunk:
                        peaks.append(round(max(abs(s) for s in chunk) / 32768.0, 3))
            os.remove(raw_audio_path)
        return {"success": True, "role": role, "filename": file.filename,
                "audio_path": audio_path, "media_path": media_path, "waveform": peaks}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _run_job(job_id: str, job: QCJobInput):
    JOBS[job_id]["status"] = "running"

    def on_progress(stage, done, total):
        JOBS[job_id]["progress"] = {"stage": stage, "done": done, "total": total}

    try:
        pipeline = QCPipeline(provider=get_provider())
        result = await pipeline.run(job, on_progress=on_progress)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result.model_dump()
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.post("/api/qc/run", status_code=202)
async def run_qc(job: QCJobInput, background_tasks: BackgroundTasks):
    try:
        get_provider()  # 키 검증 — mock 자동 폴백 없음, 실패 시 즉시 거부
    except ProviderNotConfiguredError as e:
        raise HTTPException(503, str(e))
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "progress": None, "movie": job.movie_title}
    background_tasks.add_task(_run_job, job_id, job)
    return {"job_id": job_id}


@app.get("/api/qc/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "존재하지 않는 작업입니다.")
    return JOBS[job_id]


@app.post("/api/qc/feedback")
def post_feedback(entry: FeedbackEntry):
    _feedback_store().record(entry)
    return {"ok": True}


from pydantic import BaseModel


class ReverdictRequest(BaseModel):
    excluded_finding_ids: list[str] = []


@app.post("/api/qc/jobs/{job_id}/reverdict")
def reverdict(job_id: str, req: ReverdictRequest):
    from app.core.verdict import load_config, compute_axis_scores, decide
    from app.schemas import QCFinding
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "완료된 작업이 아닙니다.")
    excluded = set(req.excluded_finding_ids)
    kept = [QCFinding(**f) for f in job["result"]["findings"] if f["id"] not in excluded]
    config = load_config()
    axis_scores = compute_axis_scores(kept, n_pairs=len(job["result"]["pairs"]), config=config)
    verdict = decide(axis_scores, kept, config)
    job["result"]["verdict"] = verdict.model_dump()  # 확정 판정으로 갱신
    return verdict.model_dump()


@app.get("/api/qc/export/{job_id}")
def export_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "완료된 작업이 아닙니다.")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["타임코드", "화자", "축", "심각도", "유형", "한국어 원문",
                     "영어 더빙", "지적 사유", "수정안", "동의 수"])
    for f in job["result"]["findings"]:
        writer.writerow([
            f"{f['start_time']:.1f}-{f['end_time']:.1f}", f["speaker"], f["axis"],
            f["severity"], f["issue_type"], f["original_text"],
            f["current_translation"], f["description"], f["recommendation"],
            f["agreement"],
        ])
    return Response(
        content="\ufeff" + buf.getvalue(),  # UTF-8 BOM — 엑셀 한글 호환
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=qc_report_{job_id}.csv"},
    )
```

- [ ] **Step 4: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/main.py backend/tests/test_api.py
git commit -m "feat: API 개편 — 잡 기반 QC 실행/진행률/피드백/CSV, translate·mock 엔드포인트 제거"
```

---

### Task 14: 프론트엔드 — API 클라이언트 + 프로젝트 뷰 (업로드/진행률)

**Files:**
- Create: `frontend/src/api.js`
- Create: `frontend/src/views/ProjectView.jsx`
- Modify: `frontend/src/App.jsx` (뷰 전환 상태 + ProjectView 연결)
- Modify: `frontend/src/App.css` (뷰 전환 탭/업로드 그리드 스타일 추가)

**Interfaces:**
- Produces:
  - `api.js`: `uploadMedia(file, role) -> Promise<{success, media_path, audio_path?, waveform?}>`, `runQC(payload) -> Promise<{job_id}>`, `getJob(jobId) -> Promise<job>`, `postFeedback(entry) -> Promise`, `exportUrl(jobId) -> string`
  - `<ProjectView uploads={…} setUploads={…} onJobComplete={(result, waveform) => …} />`
- Consumes: Task 13의 API

- [ ] **Step 1: api.js 작성**

`frontend/src/api.js`:

```javascript
const API_BASE = "http://localhost:8000";

export async function uploadMedia(file, role) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/qc/upload-media?role=${role}`, {
    method: "POST", body: form,
  });
  return res.json();
}

export async function runQC(payload) {
  const res = await fetch(`${API_BASE}/api/qc/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.status === 503) {
    const body = await res.json();
    throw new Error(body.detail);
  }
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/api/qc/jobs/${jobId}`);
  return res.json();
}

export async function postFeedback(entry) {
  const res = await fetch(`${API_BASE}/api/qc/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  return res.json();
}

export async function reverdict(jobId, excludedFindingIds) {
  const res = await fetch(`${API_BASE}/api/qc/jobs/${jobId}/reverdict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ excluded_finding_ids: excludedFindingIds }),
  });
  return res.json();
}

export function exportUrl(jobId) {
  return `${API_BASE}/api/qc/export/${jobId}`;
}
```

- [ ] **Step 2: ProjectView 작성**

`frontend/src/views/ProjectView.jsx`:

```javascript
import React, { useState, useRef } from "react";
import { uploadMedia, runQC, getJob } from "../api";

const SLOTS = [
  { role: "original", label: "한국어 원본 영상", accept: "video/*,audio/*", required: true },
  { role: "dubbed", label: "영어 더빙 완성본", accept: "video/*", required: true },
  { role: "stem", label: "다이얼로그 사운드 (스템)", accept: "audio/*", required: true },
  { role: "srt_en", label: "영어 SRT 자막", accept: ".srt", required: true },
  { role: "srt_kr", label: "한국어 SRT (선택 — 있으면 STT 생략)", accept: ".srt", required: false },
];

const STAGE_LABELS = {
  ingest: "대본 수집 (SRT/STT)", align: "타임코드 정렬",
  rules: "결정론적 룰 체크", panel: "페르소나 패널 분석", verdict: "판정 계산",
};

export default function ProjectView({ uploads, setUploads, onJobComplete }) {
  const [movieTitle, setMovieTitle] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const handleFile = async (role, file) => {
    if (!file) return;
    setUploads((u) => ({ ...u, [role]: { name: file.name, uploading: true } }));
    const res = await uploadMedia(file, role);
    setUploads((u) => ({
      ...u,
      [role]: res.success
        ? { name: file.name, ...res, uploading: false }
        : { name: file.name, error: res.error, uploading: false },
    }));
  };

  const requiredReady = SLOTS.filter((s) => s.required)
    .every((s) => uploads[s.role]?.media_path);

  const start = async () => {
    setError(null);
    setRunning(true);
    try {
      const { job_id } = await runQC({
        movie_title: movieTitle || "untitled",
        en_srt_path: uploads.srt_en.media_path,
        kr_srt_path: uploads.srt_kr?.media_path || null,
        kr_audio_path: uploads.original?.audio_path || null,
        stem_audio_path: uploads.stem?.audio_path || null,
      });
      pollRef.current = setInterval(async () => {
        const job = await getJob(job_id);
        setProgress(job.progress);
        if (job.status === "done") {
          clearInterval(pollRef.current);
          setRunning(false);
          onJobComplete(job_id, job.result);
        } else if (job.status === "error") {
          clearInterval(pollRef.current);
          setRunning(false);
          setError(job.error);
        }
      }, 2000);
    } catch (e) {
      setRunning(false);
      setError(e.message);
    }
  };

  return (
    <div className="project-view">
      <h2>새 QC 프로젝트</h2>
      <input
        className="title-input" placeholder="작품명"
        value={movieTitle} onChange={(e) => setMovieTitle(e.target.value)}
      />
      <div className="upload-grid">
        {SLOTS.map((slot) => (
          <label key={slot.role} className={`upload-slot ${uploads[slot.role]?.media_path ? "done" : ""}`}>
            <span className="slot-label">
              {slot.label}{slot.required ? " *" : ""}
            </span>
            <span className="slot-file">
              {uploads[slot.role]?.uploading ? "업로드 중…"
                : uploads[slot.role]?.name || "파일 선택"}
            </span>
            <input type="file" accept={slot.accept} hidden
              onChange={(e) => handleFile(slot.role, e.target.files[0])} />
          </label>
        ))}
      </div>
      {error && <div className="error-banner">⚠ {error}</div>}
      {running && progress && (
        <div className="progress-panel">
          <div>{STAGE_LABELS[progress.stage] || progress.stage} — {progress.done}/{progress.total}</div>
          <div className="progress-bar">
            <div className="progress-fill"
              style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }} />
          </div>
        </div>
      )}
      <button className="start-btn" disabled={!requiredReady || running} onClick={start}>
        {running ? "분석 중…" : "QC 분석 시작"}
      </button>
    </div>
  );
}
```

- [ ] **Step 3: App.jsx에 뷰 전환 연결**

`frontend/src/App.jsx` 수정:

1. 상단 import에 추가:
```javascript
import ProjectView from "./views/ProjectView";
```
2. `function App() {` 바로 아래 상태 추가:
```javascript
  const [view, setView] = useState("project"); // "project" | "review" | "report"
  const [uploads, setUploads] = useState({});
  const [jobId, setJobId] = useState(null);
  const [qcResult, setQcResult] = useState(null);
```
3. 기존 `useEffect(() => { fetchMockData(); }, [])` 를 **삭제** (mock-data 엔드포인트 제거됨).
4. 잡 완료 핸들러 추가 (`fetchMockData` 자리에):
```javascript
  const handleJobComplete = (id, result) => {
    setJobId(id);
    setQcResult(result);
    setFindings(result.findings);
    // AlignedPair → 기존 세그먼트 상태로 변환 (검수 뷰 재사용)
    setSegments(result.pairs.map((p) => ({
      id: p.id,
      start_time: (p.korean || p.dubbed).start,
      end_time: (p.korean || p.dubbed).end,
      speaker: (p.korean || p.dubbed).speaker,
      original_text: p.korean ? p.korean.text : "",
      translated_text: p.dubbed ? p.dubbed.text : "",
    })));
    updateStats(result.findings);
    setView("review");
  };
```
5. 최상위 return의 레이아웃 안쪽 상단에 뷰 전환 탭 추가, 기존 대시보드 JSX를 `view === "review"` 조건으로 감싸고 `view === "project"`일 때 `<ProjectView uploads={uploads} setUploads={setUploads} onJobComplete={handleJobComplete} />` 렌더:
```javascript
  <nav className="view-tabs">
    {[["project", "프로젝트"], ["review", "검수"], ["report", "판정/리포트"]].map(([v, label]) => (
      <button key={v} className={view === v ? "tab active" : "tab"}
        onClick={() => setView(v)}>{label}</button>
    ))}
  </nav>
```

- [ ] **Step 4: App.css에 스타일 추가** (파일 끝에)

```css
.view-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.view-tabs .tab { padding: 8px 20px; border-radius: 8px; background: transparent;
  border: 1px solid #333; color: #aaa; cursor: pointer; }
.view-tabs .tab.active { background: #1e2530; color: #fff; border-color: #4a9eff; }
.project-view { max-width: 720px; margin: 40px auto; }
.title-input { width: 100%; padding: 12px; margin-bottom: 16px; background: #12161d;
  border: 1px solid #333; border-radius: 8px; color: #fff; }
.upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.upload-slot { display: flex; flex-direction: column; gap: 6px; padding: 16px;
  border: 1px dashed #444; border-radius: 10px; cursor: pointer; }
.upload-slot.done { border-color: #3ddc84; border-style: solid; }
.slot-label { font-size: 13px; color: #8fa1b8; }
.slot-file { font-size: 14px; color: #fff; }
.progress-panel { margin: 20px 0; }
.progress-bar { height: 8px; background: #1e2530; border-radius: 4px; margin-top: 8px; }
.progress-fill { height: 100%; background: #4a9eff; border-radius: 4px; transition: width .5s; }
.start-btn { margin-top: 20px; width: 100%; padding: 14px; border-radius: 10px;
  background: #4a9eff; color: #fff; border: none; font-size: 16px; cursor: pointer; }
.start-btn:disabled { background: #2a3340; color: #667; cursor: not-allowed; }
.error-banner { margin: 16px 0; padding: 12px; border-radius: 8px;
  background: #3a1c1c; color: #ff8a8a; }
```

- [ ] **Step 5: 빌드 검증 및 수동 확인**

Run: `cd frontend && npm run build`
Expected: 빌드 성공 (에러 0)

수동 확인: `./start.sh` 실행 → http://localhost:5173 → 프로젝트 탭에 업로드 슬롯 5개(필수 4 + 선택 1) 표시, API 키 없는 상태에서 "QC 분석 시작" 시 GEMINI_API_KEY 에러 배너 표시 (mock 결과가 나오면 실패).

- [ ] **Step 6: 커밋**

```bash
git add frontend/src/api.js frontend/src/views/ProjectView.jsx frontend/src/App.jsx frontend/src/App.css
git commit -m "feat: 프로젝트 뷰 (5종 업로드 + 파이프라인 진행률) + 뷰 전환 탭"
```

---

### Task 15: 프론트엔드 — 검수 뷰 개조 (승인/반려/직접수정 + 페르소나 대안 비교)

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: `postFeedback(entry)` (Task 14), `QCFinding.alternatives`, `QCFinding.agreement`
- Produces: finding 카드의 검수 액션 3종. 모든 액션이 `POST /api/qc/feedback`으로 기록됨. `reviewedFindings: {findingId: {action, finalText}}` 상태 — Task 16의 리포트가 사용.

- [ ] **Step 1: 상태와 핸들러 교체**

`frontend/src/App.jsx`에서:

1. import에 추가: `import { postFeedback } from "./api";` (api.js에서 postFeedback를 이미 export — Task 14)
2. 상태 추가 (`qcResult` 아래):
```javascript
  const [reviewedFindings, setReviewedFindings] = useState({}); // {findingId: {action, finalText}}
```
3. 기존 `applyAIFix` 함수(App.jsx:637-654)와 `ignoreFinding` 함수(App.jsx:656-665)를 **삭제**하고 아래로 교체:
```javascript
  // 검수 액션: 모든 클릭이 피드백 저장소에 기록된다 (학습 데이터 축적 입구)
  const reviewFinding = async (finding, action, finalText = "", chosenPersona = "") => {
    await postFeedback({
      movie: qcResult?.movie_title || "untitled",
      segment_id: finding.segment_id,
      korean: finding.original_text,
      dubbed: finding.current_translation,
      finding_id: finding.id,
      reviewer_action: action, // "approved" | "rejected" | "modified"
      final_text: finalText,
      chosen_persona: chosenPersona,
    });
    setReviewedFindings((r) => ({ ...r, [finding.id]: { action, finalText } }));
  };
```
4. finding 카드 JSX에서 기존 "Apply AI Fix" / "Ignore" 버튼 영역을 찾아 (grep: `applyAIFix(` 호출부) 아래로 교체:
```javascript
  <div className="persona-alternatives">
    {Object.entries(finding.alternatives || {}).map(([persona, suggestion]) => (
      <button key={persona} className="alt-chip"
        title={`${persona}의 수정안 채택`}
        onClick={() => reviewFinding(finding, "modified", suggestion, persona)}>
        <span className="alt-persona">{persona}</span>
        <span className="alt-text">{suggestion}</span>
      </button>
    ))}
  </div>
  <div className="finding-meta">
    동의 {finding.agreement}/3 · {finding.axis} · {finding.source === "rule" ? "룰 체크" : finding.source.replace("persona:", "")}
  </div>
  <div className="review-actions">
    {reviewedFindings[finding.id] ? (
      <span className="reviewed-badge">
        {reviewedFindings[finding.id].action === "approved" ? "✓ 승인됨"
          : reviewedFindings[finding.id].action === "rejected" ? "✕ 반려됨(오탐)"
          : "✎ 수정 확정"}
      </span>
    ) : (
      <>
        <button className="btn-approve" onClick={() => reviewFinding(finding, "approved")}>승인</button>
        <button className="btn-reject" onClick={() => reviewFinding(finding, "rejected")}>반려 (오탐)</button>
        <button className="btn-modify" onClick={() => {
          const text = window.prompt("최종 영어 대사를 입력하세요:", finding.recommendation);
          if (text) reviewFinding(finding, "modified", text);
        }}>직접 수정</button>
      </>
    )}
  </div>
```
5. `handleScriptChange`는 유지 (대사 수동 편집), `fetchMockData` 관련 잔여 코드가 있으면 제거.

- [ ] **Step 2: App.css에 스타일 추가** (파일 끝에)

```css
.persona-alternatives { display: flex; flex-direction: column; gap: 6px; margin: 10px 0; }
.alt-chip { display: flex; gap: 10px; align-items: baseline; text-align: left;
  padding: 8px 12px; border-radius: 8px; background: #161c26; border: 1px solid #2a3547;
  color: #dde6f2; cursor: pointer; }
.alt-chip:hover { border-color: #4a9eff; }
.alt-persona { font-size: 11px; color: #8fa1b8; white-space: nowrap; }
.alt-text { font-size: 13px; }
.finding-meta { font-size: 12px; color: #66788f; margin: 6px 0; }
.review-actions { display: flex; gap: 8px; margin-top: 10px; }
.btn-approve { background: #1d3a2a; color: #3ddc84; border: 1px solid #2a5a3f; }
.btn-reject { background: #3a1c1c; color: #ff8a8a; border: 1px solid #5a2a2a; }
.btn-modify { background: #1e2a3f; color: #4a9eff; border: 1px solid #2a3f5f; }
.btn-approve, .btn-reject, .btn-modify { padding: 6px 14px; border-radius: 6px; cursor: pointer; }
.reviewed-badge { font-size: 13px; color: #8fa1b8; }
```

- [ ] **Step 3: 빌드 검증 및 수동 확인**

Run: `cd frontend && npm run build`
Expected: 빌드 성공

수동 확인 (mock 테스트: 백엔드를 `QC_PROVIDER` 없이 켜고 실제 키로, 또는 pytest로 확인한 API를 신뢰하고 UI만): finding 카드에 페르소나별 수정안 칩 + 동의 수 + [승인][반려][직접 수정] 버튼 표시. 버튼 클릭 후 `backend/data/feedback.jsonl`에 한 줄 추가 확인:
`cat backend/data/feedback.jsonl | tail -1`

- [ ] **Step 4: 커밋**

```bash
git add frontend/src/App.jsx frontend/src/App.css
git commit -m "feat: 검수 뷰 개조 — Apply AI Fix 제거, 승인/반려/직접수정 + 페르소나 대안 비교"
```

---

### Task 16: 프론트엔드 — 판정/리포트 뷰 (MOS 스코어카드 + 내보내기)

**Files:**
- Create: `frontend/src/views/ReportView.jsx`
- Modify: `frontend/src/App.jsx` (report 뷰 연결)
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: `qcResult.verdict` (`status`, `axis_scores[]`, `reasons[]`), `findings`, `reviewedFindings`, `exportUrl(jobId)` (Task 14)
- Produces: `<ReportView result={qcResult} jobId={jobId} findings={findings} reviewed={reviewedFindings} />`

- [ ] **Step 1: ReportView 작성**

`frontend/src/views/ReportView.jsx`:

```javascript
import React, { useState } from "react";
import { exportUrl, reverdict } from "../api";

const STATUS_META = {
  pass: { label: "통과", cls: "verdict-pass", icon: "✅" },
  conditional: { label: "조건부 통과", cls: "verdict-cond", icon: "⚠️" },
  fail: { label: "반려", cls: "verdict-fail", icon: "❌" },
};

export default function ReportView({ result, jobId, findings, reviewed }) {
  const [finalVerdict, setFinalVerdict] = useState(null);
  if (!result) return <div className="report-empty">완료된 QC 분석이 없습니다. 프로젝트 탭에서 분석을 실행하세요.</div>;

  // AI 가판정 → 검수자가 오탐을 반려한 뒤 "확정 재판정"하면 finalVerdict로 대체
  const verdict = finalVerdict || result.verdict;
  const meta = STATUS_META[verdict.status];

  const confirmVerdict = async () => {
    const excluded = Object.entries(reviewed)
      .filter(([, r]) => r.action === "rejected")
      .map(([id]) => id);
    setFinalVerdict(await reverdict(jobId, excluded));
  };
  // 검수자가 반려(오탐)한 finding 제외 = 확정 지시서
  const confirmed = findings.filter((f) => reviewed[f.id]?.action !== "rejected");

  return (
    <div className="report-view">
      <div className={`verdict-banner ${meta.cls}`}>
        <span className="verdict-icon">{meta.icon}</span>
        <span className="verdict-label">{meta.label}</span>
        <span className="verdict-note">
          AI 가판정 기준 — 검수 확정 시 오탐 제외 후 재판정됩니다.
        </span>
      </div>
      {verdict.reasons.length > 0 && (
        <ul className="verdict-reasons">
          {verdict.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}
      <h3>5축 MOS 스코어카드</h3>
      <div className="mos-grid">
        {verdict.axis_scores.map((s) => (
          <div key={s.axis} className="mos-row">
            <span className="mos-axis">{s.axis}</span>
            <span className="mos-bar">
              {[1, 2, 3, 4, 5].map((n) => (
                <span key={n} className={`mos-cell ${n <= s.mos ? `filled mos-${s.mos}` : ""}`} />
              ))}
            </span>
            <span className="mos-value">{s.mos}</span>
          </div>
        ))}
      </div>
      <h3>수정 지시서 ({confirmed.length}건)</h3>
      <table className="report-table">
        <thead>
          <tr><th>타임코드</th><th>축</th><th>심각도</th><th>원문</th><th>더빙</th><th>수정안</th><th>상태</th></tr>
        </thead>
        <tbody>
          {confirmed.map((f) => (
            <tr key={f.id}>
              <td>{f.start_time.toFixed(1)}s</td>
              <td>{f.axis}</td>
              <td className={`sev-${f.severity}`}>{f.severity}</td>
              <td>{f.original_text}</td>
              <td>{f.current_translation}</td>
              <td>{reviewed[f.id]?.finalText || f.recommendation}</td>
              <td>{reviewed[f.id] ? reviewed[f.id].action : "미검수"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="report-actions">
        <button className="export-btn" onClick={confirmVerdict}>
          검수 확정 재판정 (반려한 오탐 제외)
        </button>
        <a className="export-btn" href={exportUrl(jobId)} download>CSV 내보내기 (엑셀)</a>
        <button className="export-btn" onClick={() => window.print()}>인쇄 / PDF 저장</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: App.jsx 연결**

import 추가: `import ReportView from "./views/ReportView";`

뷰 전환부에 report 분기 추가 (Task 14의 view 조건부 렌더 위치):

```javascript
  {view === "report" && (
    <ReportView result={qcResult} jobId={jobId}
      findings={findings} reviewed={reviewedFindings} />
  )}
```

- [ ] **Step 3: App.css에 스타일 추가** (파일 끝에)

```css
.report-view { max-width: 1100px; margin: 0 auto; }
.report-empty { text-align: center; color: #66788f; margin-top: 80px; }
.verdict-banner { display: flex; align-items: center; gap: 14px; padding: 20px 24px;
  border-radius: 12px; margin-bottom: 16px; }
.verdict-pass { background: #12291c; border: 1px solid #2a5a3f; }
.verdict-cond { background: #2b2412; border: 1px solid #6a5a2a; }
.verdict-fail { background: #2b1414; border: 1px solid #5a2a2a; }
.verdict-label { font-size: 22px; font-weight: 700; }
.verdict-note { font-size: 12px; color: #8fa1b8; margin-left: auto; }
.verdict-reasons { color: #d0d8e4; margin-bottom: 24px; }
.mos-grid { display: flex; flex-direction: column; gap: 10px; margin-bottom: 32px; }
.mos-row { display: flex; align-items: center; gap: 14px; }
.mos-axis { width: 110px; color: #8fa1b8; font-size: 14px; }
.mos-bar { display: flex; gap: 4px; }
.mos-cell { width: 34px; height: 14px; border-radius: 3px; background: #1e2530; }
.mos-cell.filled.mos-5, .mos-cell.filled.mos-4 { background: #3ddc84; }
.mos-cell.filled.mos-3 { background: #e8b93d; }
.mos-cell.filled.mos-2, .mos-cell.filled.mos-1 { background: #ff6b6b; }
.mos-value { font-weight: 700; }
.report-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.report-table th, .report-table td { padding: 8px 10px; border-bottom: 1px solid #222b38;
  text-align: left; vertical-align: top; }
.sev-high { color: #ff6b6b; } .sev-medium { color: #e8b93d; } .sev-low { color: #8fa1b8; }
.report-actions { display: flex; gap: 12px; margin: 24px 0; }
.export-btn { padding: 10px 20px; border-radius: 8px; background: #1e2a3f;
  color: #4a9eff; border: 1px solid #2a3f5f; cursor: pointer; text-decoration: none; }
@media print {
  .view-tabs, .report-actions { display: none; }
  body { background: #fff; color: #000; }
}
```

- [ ] **Step 4: 빌드 검증 및 전체 수동 확인**

Run: `cd frontend && npm run build`
Expected: 빌드 성공

전체 백엔드 회귀: `cd backend && venv/bin/python -m pytest -q` → all passed

수동 E2E (실제 GEMINI_API_KEY 필요): `./start.sh` → 짧은 샘플 영상 + SRT 업로드 → 분석 → 검수 탭에서 승인/반려 → 판정/리포트 탭에서 MOS 스코어카드·CSV 다운로드 확인.

- [ ] **Step 5: 커밋**

```bash
git add frontend/src/views/ReportView.jsx frontend/src/App.jsx frontend/src/App.css
git commit -m "feat: 판정/리포트 뷰 — 5축 MOS 스코어카드, 수정 지시서, CSV/인쇄 내보내기"
```

---

## 스펙 커버리지 노트 (자체 검토 결과)

- 스펙 §3 파이프라인 ①~⑥ → Task 4·5(①), 6·7(②), 8·9(③), 10(④), 13·14·15·16(⑤), 11·15(⑥)
- 스펙 §4 mock 정책 → Task 2 (팩토리 차단) + Task 13 (503 응답)
- 스펙 §6 5축 MOS/판정 규칙 → Task 10, 리포트 표시 Task 16
- 스펙 §8 오류 처리: STT 저신뢰→정렬 신뢰도 finding(Task 6), LLM 파싱 실패 1회 재시도 후 씬 건너뜀(Task 9), API 키 부재→시작 거부(Task 2·13). **씬 청크 체크포인트 저장(중단 재개)은 v1에서 제외** — 잡이 메모리 내 단일 실행이며, 재개 요구가 실제로 생기면 후속 작업으로. (스펙과의 의도적 차이, 사용자 확인 필요)
- 스펙 §7 내보내기: CSV(엑셀 호환) + 브라우저 인쇄 기반 PDF 저장. 사내 양식 맞춤은 스펙 §10대로 범위 밖.
- 스펙 §4 context.py/voice_qc.py: 파일 유지, v1 파이프라인 미호출 (Task 12 참고)
