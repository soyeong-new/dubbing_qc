# 오디오 직접 청취 + 교차 합의 (v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 한국어 STT를 판단 경로에서 제거하고, 씬 오디오 직접 청취 + 독립 2회 청취의 의미 수준 합의로 오탐을 걸러내는 QC 파이프라인 v3를 구현한다.

**Architecture:** 영어 SRT를 타임라인 기준으로 씬을 나누고, 씬별 한국어 오디오 클립(앞 25초 맥락 포함)을 페르소나가 직접 청취한다. 페르소나 후보 지적은 별도 모델의 블라인드 재청취와 의미 수준 합의 판정을 통과해야 확정된다. 청취 불가·불일치 구간은 "판단 보류"로 커버리지에 표시한다.

**Tech Stack:** FastAPI + pydantic, google.generativeai(기존 SDK 유지), ffmpeg/ffprobe, pytest(+anyio), React(Vite)

**설계 스펙:** `docs/superpowers/specs/2026-07-20-audio-direct-consensus-design.md`

## Global Constraints

- Python은 `backend/venv/bin/python`(3.14) 사용. 테스트 실행: `cd backend && venv/bin/python -m pytest -q`
- API 요청 하나에는 **씬 하나만** 담는다 (여러 씬 오디오를 한 요청에 묶는 코드 금지)
- 씬 클립은 앞 맥락 25초 포함, mp3 32kbps 16kHz mono
- 모든 페르소나 지적에 `heard_korean`(들은 한국어) 필수, 불확실 시 지적 대신 `unheard_segment_ids`
- 룰 체크(결정론적) 지적은 합의 필터를 거치지 않고 그대로 통과한다
- `description`은 한국어, `recommendation`은 영어 (기존 규칙 유지)
- mock 프로바이더는 `PYTEST_CURRENT_TEST` 환경에서만 허용 (기존 규칙 유지)
- google.generativeai → google.genai SDK 마이그레이션은 이번 스코프 밖 (기존 SDK 유지)
- 커밋 메시지는 기존 리포 관례(한국어 `feat:`/`fix:`) 유지

---

### Task 1: 씬 클립 추출기 (`scene_clips.py`)

**Files:**
- Create: `backend/app/core/scene_clips.py`
- Test: `backend/tests/test_scene_clips.py`

**Interfaces:**
- Consumes: `app.schemas.AlignedPair`, ffmpeg CLI
- Produces: `scene_time_range(pairs: List[AlignedPair]) -> Optional[Tuple[float, float]]`,
  `extract_scene_clip(audio_path: str, start: float, end: float, ctx_seconds: float = 25.0, pad: float = 0.3, out_dir: Optional[str] = None) -> str` (mp3 경로 반환). Task 4·5가 이 두 함수를 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# backend/tests/test_scene_clips.py
import os
import subprocess

import pytest

from app.core.scene_clips import extract_scene_clip, scene_time_range
from app.schemas import AlignedPair, SegmentText


def _tone_wav(path: str, seconds: int = 40) -> None:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "16000", "-ac", "1", "-y", path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def test_extract_scene_clip_includes_context(tmp_path):
    wav = str(tmp_path / "src.wav")
    _tone_wav(wav)
    out = extract_scene_clip(wav, start=30.0, end=32.0, out_dir=str(tmp_path))
    assert os.path.exists(out) and out.endswith(".mp3")
    # 25초 맥락 + 2초 본문 + 0.3초 패드 = 27.3초
    assert abs(_duration(out) - 27.3) < 0.5


def test_extract_scene_clip_clamps_at_zero(tmp_path):
    wav = str(tmp_path / "src.wav")
    _tone_wav(wav, seconds=10)
    out = extract_scene_clip(wav, start=3.0, end=5.0, out_dir=str(tmp_path))
    # 시작이 0 밑으로 내려가지 않는다: 0~5.3초 = 5.3초
    assert abs(_duration(out) - 5.3) < 0.5


def test_scene_time_range_uses_dubbed_anchor():
    pairs = [
        AlignedPair(id="pair_1", dubbed=SegmentText(start=10.0, end=12.0, text="a")),
        AlignedPair(id="pair_2", dubbed=SegmentText(start=15.0, end=17.5, text="b")),
    ]
    assert scene_time_range(pairs) == (10.0, 17.5)


def test_scene_time_range_empty_returns_none():
    assert scene_time_range([]) is None
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_scene_clips.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.scene_clips'`

- [ ] **Step 3: 구현**

```python
# backend/app/core/scene_clips.py
"""씬 단위 오디오 클립 추출 — 판정 모델이 직접 청취할 mp3를 만든다.

맥락(ctx_seconds)을 앞에 붙이는 이유: 맥락 없는 초단문 클립은 두 모델이 같은
오류를 공유하는 상관 오류를 낳지만, 맥락을 주면 틀릴 때 서로 다르게 틀려 합의
필터가 작동한다(2026-07-20 벤치마크 실측 — 설계 스펙 §4).
"""
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

from app.schemas import AlignedPair

CTX_SECONDS = 25.0
PAD_SECONDS = 0.3


def scene_time_range(pairs: List[AlignedPair]) -> Optional[Tuple[float, float]]:
    # 영어 SRT가 타임코드 기준(주체)이다 — assign_scenes와 같은 앵커 규칙.
    anchors = [(p.dubbed or p.korean) for p in pairs if (p.dubbed or p.korean)]
    if not anchors:
        return None
    return anchors[0].start, anchors[-1].end


def extract_scene_clip(audio_path: str, start: float, end: float,
                       ctx_seconds: float = CTX_SECONDS, pad: float = PAD_SECONDS,
                       out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or tempfile.gettempdir()
    clip_start = max(0.0, start - ctx_seconds)
    duration = (end + pad) - clip_start
    out = os.path.join(out_dir, f"qc_scene_{clip_start:.3f}_{duration:.3f}.mp3")
    subprocess.run(
        ["ffmpeg", "-ss", f"{clip_start:.3f}", "-t", f"{duration:.3f}", "-i", audio_path,
         "-acodec", "libmp3lame", "-b:a", "32k", "-ar", "16000", "-ac", "1", "-y", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_scene_clips.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/scene_clips.py backend/tests/test_scene_clips.py
git commit -m "feat: 씬 단위 오디오 클립 추출기 (맥락 포함, v3 청취 입력)"
```

---

### Task 2: 스키마 확장 (heard_korean · JudgeOutput · HeldSegment · coverage)

**Files:**
- Modify: `backend/app/schemas.py`
- Test: `backend/tests/test_schemas.py` (기존 파일에 테스트 추가)

**Interfaces:**
- Produces:
  - `QCFinding.heard_korean: str = ""` — 모델이 들은 한국어 (검수자가 청취 검증에 사용)
  - `QCFinding.consensus: str = ""` — 예: `"2/2"` (합의 통과), 빈 문자열 = 미확정
  - `HeldSegment(scene_id: str, segment_id: str = "", start: float, end: float, reason: str)`
  - `JudgeOutput(findings: List[QCFinding], unheard_segment_ids: List[str])`
  - `QCResult.held: List[HeldSegment] = []`

- [ ] **Step 1: 실패하는 테스트 추가**

`backend/tests/test_schemas.py` 파일 끝에 추가:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ImportError: cannot import name 'HeldSegment'`

- [ ] **Step 3: 구현**

`backend/app/schemas.py`의 `QCFinding`에 필드 2개 추가 (`finding_type` 아래):

```python
    heard_korean: str = ""   # 모델이 실제로 들은 한국어 — 검수자가 청취 정확성을 검증하는 고리
    consensus: str = ""      # "2/2" 등 합의 수준. 빈 문자열 = 합의 미실시(룰 체크 등)
```

`AxisScore` 클래스 위(또는 `QCFinding` 아래)에 신규 모델 2개 추가:

```python
class HeldSegment(BaseModel):
    """판단 보류 구간 — 청취 불가/교차 불일치로 검증하지 못한 커버리지 공백."""
    scene_id: str
    segment_id: str = ""
    start: float
    end: float
    reason: str  # "청취 불가" | "교차 불일치"


class JudgeOutput(BaseModel):
    """페르소나 1회 호출의 결과: 지적 + 정직한 보류 목록."""
    findings: List["QCFinding"] = Field(default_factory=list)
    unheard_segment_ids: List[str] = Field(default_factory=list)
```

`QCResult`에 필드 추가:

```python
class QCResult(BaseModel):
    verdict: Verdict
    findings: List[QCFinding]
    pairs: List[AlignedPair]
    held: List[HeldSegment] = Field(default_factory=list)
```

- [ ] **Step 4: 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_schemas.py -v`
Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/schemas.py backend/tests/test_schemas.py
git commit -m "feat: v3 스키마 — heard_korean/consensus/JudgeOutput/HeldSegment/coverage"
```

---

### Task 3: Persona 오디오 플래그 분리 + judge 반환형 전환

**Files:**
- Modify: `backend/app/providers/base.py` (Persona 플래그, judge 시그니처, `complete_json` 추상 메서드)
- Modify: `backend/app/providers/mock.py` (JudgeOutput 반환 + complete_json)
- Modify: `backend/app/core/judge_panel.py` (PERSONAS 플래그만 — run_panel 배선은 Task 5)
- Test: `backend/tests/test_providers.py` (수정)

**Interfaces:**
- Consumes: Task 2의 `JudgeOutput`
- Produces:
  - `Persona.uses_kr_audio: bool = False`, `Persona.uses_en_audio: bool = False` (`uses_audio` 제거)
  - `ModelProvider.judge(...) -> JudgeOutput` (기존 파라미터명 유지: `audio_clip_path`=영어 더빙, `original_audio_clip_path`=한국어 원본)
  - `ModelProvider.complete_json(prompt: str) -> dict` (async abstract) — Task 6 합의 판정용
  - PERSONAS: culture=`uses_kr_audio`, native=오디오 없음, director=둘 다

- [ ] **Step 1: 실패하는 테스트 수정·추가**

`backend/tests/test_providers.py`에서 mock judge 호출부를 찾아 반환형 검증을 교체하고, 아래 테스트를 추가한다 (기존 `await provider.judge(...)` 결과를 리스트로 쓰는 단언은 `.findings`로 바꾼다):

```python
import pytest

from app.providers.mock import MockProvider
from app.providers.base import Persona
from app.schemas import AlignedPair, JudgeOutput, SegmentText


@pytest.mark.anyio
async def test_mock_judge_returns_judge_output():
    pairs = [AlignedPair(
        id="pair_1",
        korean=SegmentText(start=0, end=1, text="어이가 없네"),
        dubbed=SegmentText(start=0, end=1, text="My kidney hurts"),
    )]
    persona = Persona(key="culture", name="n", instruction="i")
    out = await MockProvider().judge(pairs, persona, "")
    assert isinstance(out, JudgeOutput)
    assert out.findings and out.findings[0].segment_id == "pair_1"
    assert out.unheard_segment_ids == []


@pytest.mark.anyio
async def test_mock_complete_json_agrees():
    res = await MockProvider().complete_json("아무 프롬프트")
    assert res == {"same_issue": True}


def test_persona_audio_flags_default_off():
    p = Persona(key="k", name="n", instruction="i")
    assert p.uses_kr_audio is False and p.uses_en_audio is False
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_providers.py -v`
Expected: FAIL — `JudgeOutput` 반환 아님 / `complete_json` 없음 / 플래그 없음

- [ ] **Step 3: base.py 수정**

```python
# backend/app/providers/base.py — Persona와 ModelProvider 교체
class Persona(BaseModel):
    key: str
    name: str
    instruction: str
    uses_kr_audio: bool = False   # 한국어 원본 오디오를 직접 청취하는가
    uses_en_audio: bool = False   # 영어 더빙 스템 오디오를 청취하는가
    axes: List[str] = Field(default_factory=list)


class ModelProvider(ABC):
    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> "JudgeOutput":
        ...

    @abstractmethod
    async def complete_json(self, prompt: str) -> dict:
        """텍스트 프롬프트 1개를 JSON 응답으로 — 합의 의미 일치 판정 등 경량 호출용."""
        ...
```

import에 `JudgeOutput` 추가: `from app.schemas import AlignedPair, JudgeOutput, QCFinding`

- [ ] **Step 4: mock.py 수정**

`MockProvider.judge`의 마지막 `return findings`를 다음으로 교체하고 메서드 추가:

```python
        return JudgeOutput(findings=findings)

    async def complete_json(self, prompt: str) -> dict:
        # 결정론적: 합의 판정은 항상 동의. 불일치 경로는 테스트가 FakeProvider를 주입해 검증한다.
        return {"same_issue": True}
```

import 갱신: `from app.schemas import AlignedPair, JudgeOutput, QCFinding`

- [ ] **Step 5: judge_panel.py PERSONAS 플래그 교체**

`PERSONAS` 정의에서:
- culture 페르소나에 `uses_kr_audio=True,` 추가
- director 페르소나의 `uses_audio=True,`를 `uses_kr_audio=True, uses_en_audio=True,`로 교체
- native는 플래그 없음 (텍스트만)

culture instruction 끝에 한 문장 추가:

```python
            "한국어 원문은 오디오로 제공됩니다 — 직접 듣고 판단하십시오."
```

`run_panel` 안의 `persona.uses_audio` 참조 2곳을 임시로 새 플래그에 맞춘다 (완전한 배선은 Task 5):

```python
            audio = clip_path if persona.uses_en_audio else None
            orig_audio = original_clip_path if persona.uses_kr_audio else None
```

`run_panel`의 `all_findings.extend(...)` 줄을 JudgeOutput에 맞게 교체:

```python
                    out = await provider.judge(
                        pairs, persona, knowledge,
                        audio_clip_path=audio, original_audio_clip_path=orig_audio,
                    )
                    all_findings.extend(out.findings)
```

- [ ] **Step 6: 전체 테스트 실행, 깨지는 곳 정리**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: `test_judge_panel.py`·`test_gemini_provider.py`·`test_pipeline.py`에서 `uses_audio`/반환형 관련 실패 가능. 다음 규칙으로 고친다:
- 테스트 코드의 `uses_audio=True` → `uses_kr_audio=True, uses_en_audio=True`
- 테스트에서 judge 반환 리스트 단언 → `.findings` 단언
- `GeminiProvider.judge`는 Task 4에서 교체하므로, gemini 테스트가 반환형으로 실패하면 해당 단언만 `.findings`로 바꾸되 프롬프트 관련 실패는 Task 4에서 다룬다 (여기서 실패가 남으면 `@pytest.mark.skip(reason="Task 4에서 v3 프롬프트로 교체")`를 임시로 달고 Task 4에서 제거)

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: passed (또는 Task 4 예정 skip만 남음)

- [ ] **Step 7: 커밋**

```bash
git add backend/app/providers/base.py backend/app/providers/mock.py backend/app/core/judge_panel.py backend/tests
git commit -m "feat: 페르소나 오디오 플래그 분리(kr/en) + judge가 JudgeOutput 반환"
```

---

### Task 4: Gemini 프롬프트 v3 + 모델 풀/429 백오프

**Files:**
- Modify: `backend/app/providers/gemini.py`
- Test: `backend/tests/test_gemini_provider.py`

**Interfaces:**
- Consumes: Task 2 `JudgeOutput`, Task 3 Persona 플래그
- Produces:
  - `GeminiProvider(model_pool: Optional[List[str]] = None)` — env `QC_GEMINI_MODELS`(콤마 구분, 기본 `"gemini-3.5-flash,gemini-3.1-flash-lite"`)
  - `GeminiProvider._generate(parts) -> str` — 429 시 15s·30s 대기 후 재시도, 소진 시 풀의 다음 모델로 (무료 저속 모드)
  - `build_judge_prompt_v3(pairs, persona, knowledge) -> str`
  - `parse_judge_response_v3(text, pairs, persona) -> JudgeOutput`
  - `GeminiProvider.complete_json(prompt) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_gemini_provider.py`를 열어 기존 `build_judge_prompt`/`parse_judge_response` 테스트를 v3로 교체·추가한다:

```python
import json

import pytest

from app.providers.gemini import build_judge_prompt_v3, parse_judge_response_v3
from app.providers.base import Persona
from app.schemas import AlignedPair, SegmentText


def _pairs():
    return [AlignedPair(
        id="pair_1", scene_id="scene_1",
        korean=None,  # v3: 한국어는 오디오로만 제공되는 경우
        dubbed=SegmentText(start=10.0, end=12.0, speaker="A", text="Have you eaten?"),
    )]


def _persona():
    return Persona(key="culture", name="한국 문화·언어 전문가",
                   instruction="검사하라", uses_kr_audio=True, axes=["언어 적합성"])


def test_prompt_v3_mentions_audio_and_heard_korean():
    prompt = build_judge_prompt_v3(_pairs(), _persona(), "")
    assert "heard_korean" in prompt
    assert "unheard_segment_ids" in prompt
    assert "오디오" in prompt
    assert "Have you eaten?" in prompt


def test_parse_v3_returns_findings_and_unheard():
    raw = json.dumps({
        "findings": [{
            "segment_id": "pair_1", "heard_korean": "밥 먹었어?",
            "severity": "medium", "issue_type": "문화적 정서 차이",
            "description": "안부 인사가 직역되었습니다.",
            "recommendation": "How have you been?",
            "confidence": 0.9, "axis": "언어 적합성", "finding_type": "quality",
        }],
        "unheard_segment_ids": ["pair_99", "pair_1x"],
    })
    out = parse_judge_response_v3(raw, _pairs(), _persona())
    f = out.findings[0]
    assert f.heard_korean == "밥 먹었어?"
    # korean 텍스트가 없으면 들은 내용이 원문 표시로 쓰인다
    assert f.original_text == "밥 먹었어?"
    assert f.start_time == 10.0  # 앵커는 영어 SRT
    assert out.unheard_segment_ids == ["pair_99", "pair_1x"]


def test_parse_v3_ignores_unknown_segment():
    raw = json.dumps({"findings": [{"segment_id": "no_such", "heard_korean": "x",
                                    "severity": "low", "issue_type": "t",
                                    "description": "d", "recommendation": "r",
                                    "confidence": 0.5}], "unheard_segment_ids": []})
    out = parse_judge_response_v3(raw, _pairs(), _persona())
    assert out.findings == []


@pytest.mark.anyio
async def test_generate_falls_back_to_next_model_on_429(monkeypatch):
    from app.providers import gemini as g

    calls = []

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def generate_content(self, parts, generation_config=None):
            calls.append(self.name)
            if self.name == "m1":
                raise RuntimeError("429 You exceeded your current quota")

            class R:
                text = '{"ok": true}'
            return R()

    class FakeGenai:
        @staticmethod
        def GenerativeModel(name):
            return FakeModel(name)

    monkeypatch.setattr(g.asyncio, "sleep", _instant_sleep())
    provider = g.GeminiProvider.__new__(g.GeminiProvider)
    provider._genai = FakeGenai()
    provider.model_pool = ["m1", "m2"]
    text = await provider._generate(["prompt"])
    assert text == '{"ok": true}'
    assert calls == ["m1", "m1", "m1", "m2"]  # m1 3회 재시도 후 m2 폴백


def _instant_sleep():
    async def _sleep(_secs):
        return None
    return _sleep
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py -v`
Expected: FAIL — `build_judge_prompt_v3` 미정의 등

- [ ] **Step 3: gemini.py 구현**

기존 `JUDGE_PROMPT_TEMPLATE`/`build_judge_prompt`/`parse_judge_response`는 삭제하고 아래로 교체한다. `MODEL_NAME` 상수도 삭제한다.

```python
DEFAULT_MODEL_POOL = "gemini-3.5-flash,gemini-3.1-flash-lite"

JUDGE_PROMPT_TEMPLATE_V3 = """
당신은 한국 영화의 영어 더빙을 검수하는 "{persona_name}"입니다.

{persona_instruction}

## 검수 지식베이스 (참고 규칙)
{knowledge}

## 지시
한국어 원본이 제공되는 경우 텍스트가 아니라 **오디오**입니다. 씬 오디오 앞부분에는
직전 장면 맥락이 포함되어 있습니다 — 아래 나열된 세그먼트 시간 범위의 발화만
평가 대상입니다.
- 각 지적에는 heard_korean(그 세그먼트에서 실제로 들은 한국어 대사)을 반드시 포함하십시오.
- 발화가 안 들리거나 내용에 확신이 없는 세그먼트는 **지적하지 말고**
  unheard_segment_ids 배열에 넣으십시오. 지어낸 지적보다 정직한 보류가 훨씬 좋습니다.
- description은 반드시 **한국어**, recommendation은 반드시 **영어** 더빙 대사.
- axis는 다음 중 하나: {axes}
- severity는 "high" | "medium" | "low", finding_type은 "quality" 또는 "sensitive".
- 문제 없는 세그먼트는 findings에 포함하지 마십시오.

반환 스키마:
{{"findings": [{{"segment_id": "...", "heard_korean": "...", "severity": "...",
  "issue_type": "...", "description": "...", "recommendation": "...",
  "confidence": 0.9, "axis": "...", "finding_type": "quality"}}],
 "unheard_segment_ids": ["pair_7"]}}

## 세그먼트 (영어 더빙 대사와 시간 범위)
{payload}
"""


def build_judge_prompt_v3(pairs: List[AlignedPair], persona: Persona, knowledge: str) -> str:
    payload = []
    for p in pairs:
        anchor = p.dubbed or p.korean
        item = {
            "segment_id": p.id,
            "english_dub": p.dubbed.text if p.dubbed else "",
            "speaker": anchor.speaker if anchor else "?",
            "start": anchor.start if anchor else 0,
            "end": anchor.end if anchor else 0,
        }
        # 한국어 SRT가 있고 이 페르소나가 오디오를 안 듣는 경우에만 텍스트 제공
        if p.korean and not persona.uses_kr_audio:
            item["korean"] = p.korean.text
        payload.append(item)
    return JUDGE_PROMPT_TEMPLATE_V3.format(
        persona_name=persona.name,
        persona_instruction=persona.instruction,
        knowledge=knowledge or "(등록된 규칙 없음)",
        axes=" | ".join(persona.axes or AXES),
        payload=json.dumps(payload, ensure_ascii=False, indent=1),
    )


def parse_judge_response_v3(text: str, pairs: List[AlignedPair],
                            persona: Persona) -> JudgeOutput:
    by_id = {p.id: p for p in pairs}
    default_axis = persona.axes[0] if persona.axes else "언어 적합성"
    obj = json.loads(text)
    findings = []
    for i, item in enumerate(obj.get("findings", [])):
        pair = by_id.get(item.get("segment_id"))
        if pair is None:
            continue
        axis = item.get("axis", default_axis)
        if axis not in AXES:
            axis = default_axis
        finding_type = item.get("finding_type", "quality")
        if finding_type not in ("quality", "sensitive"):
            finding_type = "quality"
        heard = item.get("heard_korean", "")
        # 영어 SRT가 타임코드 기준(주체)이다 (기존 원칙 유지)
        anchor = pair.dubbed or pair.korean
        findings.append(QCFinding(
            id=f"{persona.key}_{pair.id}_{i}",
            segment_id=pair.id,
            category="localization",
            severity=item.get("severity", "medium"),
            issue_type=item.get("issue_type", "번역 오류"),
            start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
            description=item.get("description", ""),
            original_text=pair.korean.text if pair.korean else heard,
            current_translation=pair.dubbed.text if pair.dubbed else "",
            recommendation=item.get("recommendation", ""),
            confidence=float(item.get("confidence", 0.8)),
            axis=axis,
            source=f"persona:{persona.key}",
            finding_type=finding_type,
            heard_korean=heard,
        ))
    return JudgeOutput(findings=findings,
                       unheard_segment_ids=list(obj.get("unheard_segment_ids", [])))
```

`GeminiProvider`를 교체한다:

```python
class GeminiProvider(ModelProvider):
    def __init__(self, model_pool: Optional[List[str]] = None):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._genai = genai
        pool = model_pool or os.getenv("QC_GEMINI_MODELS", DEFAULT_MODEL_POOL).split(",")
        self.model_pool = [m.strip() for m in pool if m.strip()]

    async def _generate(self, parts) -> str:
        """모델 풀 순회 + 429 백오프. 쿼터 소진 시 다음 모델로 폴백(무료 저속 모드).

        429 백오프 코드는 유료 전환 후에도 안전장치로 그대로 쓰인다(설계 스펙 §5).
        """
        last_err: Optional[Exception] = None
        for model_name in self.model_pool:
            model = self._genai.GenerativeModel(model_name)
            for attempt in range(3):
                try:
                    response = await asyncio.to_thread(
                        model.generate_content, parts,
                        generation_config={"response_mime_type": "application/json"},
                    )
                    return response.text
                except Exception as e:
                    last_err = e
                    if "429" in str(e):
                        await asyncio.sleep(15 * (attempt + 1))
                    else:
                        break  # 429가 아니면 이 모델 재시도 무의미 — 다음 모델로
        raise last_err

    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> JudgeOutput:
        prompt = build_judge_prompt_v3(pairs, persona, knowledge)
        parts = [prompt]
        if audio_clip_path and persona.uses_en_audio and os.path.exists(audio_clip_path):
            with open(audio_clip_path, "rb") as f:
                parts.insert(0, {"mime_type": "audio/mp3", "data": f.read()})
            parts.insert(0, "[다음 오디오는 영어 더빙입니다]")
        if original_audio_clip_path and persona.uses_kr_audio and os.path.exists(original_audio_clip_path):
            with open(original_audio_clip_path, "rb") as f:
                parts.insert(0, {"mime_type": "audio/mp3", "data": f.read()})
            parts.insert(0, "[다음 오디오는 한국어 원본입니다 (앞부분은 맥락)]")
        text = await self._generate(parts)
        return parse_judge_response_v3(text, pairs, persona)

    async def complete_json(self, prompt: str) -> dict:
        return json.loads(await self._generate([prompt]))
```

import 갱신: `from app.schemas import AlignedPair, JudgeOutput, QCFinding, AXES`
참고: 씬 클립은 Task 1 추출기가 이미 mp3로 만들므로 `_compress_to_mp3`는 삭제한다.

- [ ] **Step 4: 통과 확인 (Task 3의 임시 skip 제거 포함)**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py -v && venv/bin/python -m pytest -q`
Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/providers/gemini.py backend/tests/test_gemini_provider.py backend/tests
git commit -m "feat: Gemini v3 — 오디오 직접 청취 프롬프트(heard_korean/보류) + 모델 풀 429 폴백"
```

---

### Task 5: run_panel 씬 클립 배선 + 보류 수집

**Files:**
- Modify: `backend/app/core/judge_panel.py`
- Test: `backend/tests/test_judge_panel.py`

**Interfaces:**
- Consumes: Task 1 `extract_scene_clip`/`scene_time_range`, Task 2 `HeldSegment`
- Produces: `run_panel(scenes, knowledge, provider, stem_wav_path=None, kr_audio_path=None, on_progress=None) -> Tuple[List[QCFinding], List[HeldSegment]]`
  - 반환 1: merge_findings 적용된 페르소나 지적 (후보)
  - 반환 2: unheard 세그먼트의 HeldSegment 목록 (segment_id 기준 중복 제거)

- [ ] **Step 1: 실패하는 테스트 수정**

`backend/tests/test_judge_panel.py`에서 `run_panel` 반환을 tuple로 받도록 기존 테스트를 수정하고(`findings = await run_panel(...)` → `findings, held = await run_panel(...)`), 보류 수집 테스트를 추가한다:

```python
import pytest

from app.core.judge_panel import run_panel
from app.providers.base import ModelProvider, Persona
from app.schemas import AlignedPair, HeldSegment, JudgeOutput, SegmentText


class UnheardProvider(ModelProvider):
    async def judge(self, pairs, persona, knowledge, audio_clip_path=None,
                    original_audio_clip_path=None):
        return JudgeOutput(findings=[], unheard_segment_ids=[pairs[0].id])

    async def complete_json(self, prompt):
        return {"same_issue": True}


@pytest.mark.anyio
async def test_run_panel_collects_unheard_as_held():
    pairs = [AlignedPair(id="pair_1", scene_id="scene_1",
                         dubbed=SegmentText(start=1.0, end=2.0, text="hello"))]
    findings, held = await run_panel({"scene_1": pairs}, "", UnheardProvider())
    assert findings == []
    # 페르소나 3명이 모두 보류해도 세그먼트당 1건으로 중복 제거된다
    assert len(held) == 1
    assert held[0] == HeldSegment(scene_id="scene_1", segment_id="pair_1",
                                  start=1.0, end=2.0, reason="청취 불가")
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_judge_panel.py -v`
Expected: FAIL — 반환형 불일치

- [ ] **Step 3: run_panel 교체**

```python
async def run_panel(scenes: Dict[str, List[AlignedPair]], knowledge: str,
                    provider: ModelProvider, stem_wav_path: Optional[str] = None,
                    kr_audio_path: Optional[str] = None,
                    on_progress: Optional[Callable[[int, int], None]] = None,
                    ) -> Tuple[List[QCFinding], List[HeldSegment]]:
    from app.core.scene_clips import extract_scene_clip, scene_time_range

    all_findings: List[QCFinding] = []
    held_by_segment: Dict[str, HeldSegment] = {}
    scene_ids = sorted(scenes.keys(), key=lambda s: int(s.split("_")[1]))
    for done, scene_id in enumerate(scene_ids, start=1):
        pairs = scenes[scene_id]
        pair_by_id = {p.id: p for p in pairs}
        time_range = scene_time_range(pairs)
        kr_clip = None
        en_clip = None
        if time_range:
            start, end = time_range
            if kr_audio_path:
                try:
                    # 한국어 원본: 앞 25초 맥락 포함 (합의 필터의 전제조건 — 스펙 §2)
                    kr_clip = await asyncio.to_thread(
                        extract_scene_clip, kr_audio_path, start, end)
                except Exception as e:
                    print(f"[패널] {scene_id} 한국어 클립 추출 실패, 오디오 없이 진행: {e}")
            if stem_wav_path:
                try:
                    # 영어 더빙: 평가 대상 구간만 (맥락 불필요)
                    en_clip = await asyncio.to_thread(
                        extract_scene_clip, stem_wav_path, start, end, 0.0)
                except Exception as e:
                    print(f"[패널] {scene_id} 더빙 클립 추출 실패, 오디오 없이 진행: {e}")
        for persona in PERSONAS:
            audio = en_clip if persona.uses_en_audio else None
            orig_audio = kr_clip if persona.uses_kr_audio else None
            for attempt in (1, 2):
                try:
                    out = await provider.judge(
                        pairs, persona, knowledge,
                        audio_clip_path=audio, original_audio_clip_path=orig_audio,
                    )
                    all_findings.extend(out.findings)
                    for seg_id in out.unheard_segment_ids:
                        pair = pair_by_id.get(seg_id)
                        if pair is None or seg_id in held_by_segment:
                            continue
                        anchor = pair.dubbed or pair.korean
                        held_by_segment[seg_id] = HeldSegment(
                            scene_id=scene_id, segment_id=seg_id,
                            start=anchor.start, end=anchor.end, reason="청취 불가")
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[패널] {scene_id}/{persona.key} 분석 실패 (2회 시도): {e}")
        if on_progress:
            on_progress(done, len(scene_ids))
    return merge_findings(all_findings), list(held_by_segment.values())
```

import 갱신: `from typing import Callable, Dict, List, Optional, Tuple` /
`from app.schemas import AlignedPair, HeldSegment, QCFinding`
(기존 `from app.core.rule_checks import extract_clip` 로컬 import는 삭제)

- [ ] **Step 4: 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_judge_panel.py -v && venv/bin/python -m pytest -q`
Expected: test_pipeline이 run_panel 반환형으로 실패할 수 있음 — pipeline은 Task 7에서 배선하므로 여기서는 test_pipeline의 호출부만 `panel_findings, _held = await run_panel(...)` 형태로 임시 수정하지 말고, `pipeline.py`의 호출부를 아래처럼 최소 수정한다:

```python
        panel_findings, held = await run_panel(
            scenes, load_knowledge(), provider,
            stem_wav_path=job.stem_audio_path,
            kr_audio_path=job.kr_audio_path,
            on_progress=lambda d, t: notify("panel", d, t),
        )
        findings += panel_findings
```

그리고 `QCResult(verdict=verdict, findings=findings, pairs=pairs, held=held)`로 반환을 바꾼다.

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/judge_panel.py backend/app/core/pipeline.py backend/tests/test_judge_panel.py
git commit -m "feat: 패널이 씬 클립(맥락 포함) 직접 청취 + 청취 불가를 판단 보류로 수집"
```

---

### Task 6: 합의 필터 (`consensus.py`)

**Files:**
- Create: `backend/app/core/consensus.py`
- Modify: `backend/app/providers/base.py` (`get_secondary_provider` 추가)
- Test: `backend/tests/test_consensus.py`

**Interfaces:**
- Consumes: Task 1 `extract_scene_clip`/`scene_time_range`, Task 3 `ModelProvider.judge`/`complete_json`, Task 2 스키마
- Produces:
  - `confirm_findings(candidates: List[QCFinding], scenes: Dict[str, List[AlignedPair]], kr_audio_path: Optional[str], provider: ModelProvider, knowledge: str = "") -> Tuple[List[QCFinding], List[HeldSegment]]`
    - 확정 지적: `consensus="2/2"`, `agreement=2`로 갱신되어 반환
    - `kr_audio_path`가 None이면 합의 절차 없이 원본 그대로 반환 (오디오 없는 구성의 우아한 저하)
  - `get_secondary_provider() -> ModelProvider` — env `QC_GEMINI_MODELS_SECONDARY`(기본은 기본 풀의 역순). mock 모드에서는 MockProvider.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# backend/tests/test_consensus.py
import pytest

from app.core.consensus import confirm_findings, VERIFIER_PERSONA
from app.providers.base import ModelProvider
from app.schemas import AlignedPair, JudgeOutput, QCFinding, SegmentText


def _pair(pid="pair_1"):
    return AlignedPair(id=pid, scene_id="scene_1",
                       dubbed=SegmentText(start=1.0, end=2.0, text="hello"))


def _candidate(pid="pair_1"):
    return QCFinding(
        id=f"culture_{pid}_0", segment_id=pid, category="localization",
        severity="medium", issue_type="번역 오류", start_time=1.0, end_time=2.0,
        speaker="?", description="원문 뉘앙스 소실", original_text="들은 한국어",
        current_translation="hello", recommendation="hey", confidence=0.9,
        source="persona:culture", heard_korean="들은 한국어")


class FakeProvider(ModelProvider):
    """블라인드 재청취가 지적을 내고, 의미 일치 판정이 same_issue를 반환하는 더블."""

    def __init__(self, blind_findings, same_issue):
        self._blind = blind_findings
        self._same = same_issue
        self.judge_calls = 0

    async def judge(self, pairs, persona, knowledge, audio_clip_path=None,
                    original_audio_clip_path=None):
        self.judge_calls += 1
        assert persona.key == VERIFIER_PERSONA.key  # 블라인드 검증자만 호출되어야 함
        return JudgeOutput(findings=self._blind)

    async def complete_json(self, prompt):
        return {"same_issue": self._same}


@pytest.mark.anyio
async def test_agreed_finding_is_confirmed(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    blind = [_candidate()]
    provider = FakeProvider(blind, same_issue=True)
    confirmed, held = await confirm_findings(
        [_candidate()], {"scene_1": [_pair()]}, "/fake/audio.wav", provider)
    assert len(confirmed) == 1
    assert confirmed[0].consensus == "2/2" and confirmed[0].agreement == 2
    assert held == []


@pytest.mark.anyio
async def test_disagreed_finding_is_dropped(monkeypatch):
    _no_ffmpeg(monkeypatch)
    provider = FakeProvider([], same_issue=False)
    confirmed, held = await confirm_findings(
        [_candidate()], {"scene_1": [_pair()]}, "/fake/audio.wav", provider)
    assert confirmed == []
    # 불일치는 폐기이지 보류가 아니다 (스펙 §2 결정 2)
    assert held == []


@pytest.mark.anyio
async def test_verifier_unheard_becomes_held(monkeypatch):
    _no_ffmpeg(monkeypatch)

    class UnheardProvider(FakeProvider):
        async def judge(self, pairs, persona, knowledge, audio_clip_path=None,
                        original_audio_clip_path=None):
            return JudgeOutput(findings=[], unheard_segment_ids=["pair_1"])

    confirmed, held = await confirm_findings(
        [_candidate()], {"scene_1": [_pair()]}, "/fake/audio.wav",
        UnheardProvider([], True))
    assert confirmed == []
    assert len(held) == 1 and held[0].reason == "청취 불가"


@pytest.mark.anyio
async def test_no_audio_passes_through():
    provider = FakeProvider([], same_issue=False)
    cand = [_candidate()]
    confirmed, held = await confirm_findings(cand, {"scene_1": [_pair()]}, None, provider)
    assert confirmed == cand and held == []
    assert provider.judge_calls == 0  # 오디오 없으면 재청취 자체를 안 한다


def _no_ffmpeg(monkeypatch):
    # 합의 로직 테스트에서 실제 ffmpeg 클립 추출을 우회한다
    from app.core import consensus
    monkeypatch.setattr(consensus, "extract_scene_clip",
                        lambda *a, **k: "/fake/clip.mp3")
```

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_consensus.py -v`
Expected: FAIL — 모듈 없음

- [ ] **Step 3: consensus.py 구현**

```python
# backend/app/core/consensus.py
"""교차 합의 필터 — 페르소나 후보 지적을 다른 모델의 블라인드 재청취로 확정한다.

첫 번째 모델이 뭘 지적했는지 알려주지 않는 것(블라인드)이 핵심이다 — 알려주면
동조 편향으로 교차검증이 무의미해진다(설계 스펙 §2). 단독 모델 정확도는 60%
수준(벤치마크 실측)이므로 이 필터는 옵션이 아니라 필수 부품이다.
"""
import asyncio
import json
from typing import Dict, List, Optional, Tuple

from app.core.scene_clips import extract_scene_clip, scene_time_range
from app.providers.base import ModelProvider, Persona
from app.schemas import AlignedPair, HeldSegment, QCFinding

VERIFIER_PERSONA = Persona(
    key="verifier", name="블라인드 재검수자",
    axes=["언어 적합성", "자연스러움", "감정 표현"], uses_kr_audio=True,
    instruction=(
        "당신은 이 작품을 처음 보는 독립 검수자입니다. 한국어 원본 오디오를 듣고 "
        "영어 더빙 대사와 대조하여, 로컬라이제이션 문제가 있는 세그먼트만 "
        "지적하십시오. 다른 검수자의 의견은 제공되지 않으며, 문제가 없어 보이면 "
        "지적하지 않는 것이 정답일 수 있습니다."
    ),
)

MATCH_PROMPT = """두 검수 의견이 같은 문제를 지적하는지 판정하십시오.

## 의견 A (세그먼트 {segment_id})
- 들은 한국어: {a_heard}
- 지적: {a_description}

## 의견 B의 지적 목록 (같은 씬)
{b_findings}

의견 B 목록 중 하나라도 의견 A와 **같은 세그먼트의 같은 종류 문제**를 지적하면
same_issue를 true로 하십시오. 표현이 달라도 문제의 본질(무엇이 왜 잘못됐는가)이
같으면 같은 문제입니다.

JSON으로만 반환: {{"same_issue": true}} 또는 {{"same_issue": false}}"""


async def confirm_findings(candidates: List[QCFinding],
                           scenes: Dict[str, List[AlignedPair]],
                           kr_audio_path: Optional[str],
                           provider: ModelProvider,
                           knowledge: str = "",
                           ) -> Tuple[List[QCFinding], List[HeldSegment]]:
    if not candidates:
        return [], []
    if not kr_audio_path:
        # 오디오 없는 구성(예: 텍스트만 있는 회귀 테스트)에서는 합의를 건너뛴다 —
        # 확정 없는 원본 그대로 반환 (consensus 빈 문자열 = 미확정 표시)
        return candidates, []

    pair_scene: Dict[str, str] = {p.id: sid for sid, ps in scenes.items() for p in ps}
    by_scene: Dict[str, List[QCFinding]] = {}
    for f in candidates:
        by_scene.setdefault(pair_scene.get(f.segment_id, ""), []).append(f)

    confirmed: List[QCFinding] = []
    held: List[HeldSegment] = []
    for scene_id, cands in by_scene.items():
        pairs = scenes.get(scene_id, [])
        time_range = scene_time_range(pairs)
        if not pairs or not time_range:
            continue
        try:
            clip = await asyncio.to_thread(
                extract_scene_clip, kr_audio_path, time_range[0], time_range[1])
        except Exception as e:
            print(f"[합의] {scene_id} 클립 추출 실패, 이 씬 후보는 미확정 유지: {e}")
            confirmed.extend(cands)
            continue
        blind = await provider.judge(pairs, VERIFIER_PERSONA, knowledge,
                                     original_audio_clip_path=clip)
        unheard = set(blind.unheard_segment_ids)
        b_list = [
            {"segment_id": bf.segment_id, "heard_korean": bf.heard_korean,
             "description": bf.description}
            for bf in blind.findings
        ]
        pair_by_id = {p.id: p for p in pairs}
        for cand in cands:
            if cand.segment_id in unheard:
                anchor_pair = pair_by_id.get(cand.segment_id)
                anchor = (anchor_pair.dubbed or anchor_pair.korean) if anchor_pair else None
                held.append(HeldSegment(
                    scene_id=scene_id, segment_id=cand.segment_id,
                    start=anchor.start if anchor else cand.start_time,
                    end=anchor.end if anchor else cand.end_time,
                    reason="청취 불가"))
                continue
            same_scene_b = [b for b in b_list if b["segment_id"] == cand.segment_id]
            if not same_scene_b:
                continue  # 재청취자가 같은 세그먼트를 지적하지 않음 → 불일치 → 폐기
            res = await provider.complete_json(MATCH_PROMPT.format(
                segment_id=cand.segment_id, a_heard=cand.heard_korean,
                a_description=cand.description,
                b_findings=json.dumps(same_scene_b, ensure_ascii=False, indent=1)))
            if res.get("same_issue") is True:
                confirmed.append(cand.model_copy(
                    update={"consensus": "2/2", "agreement": 2}))
    confirmed.sort(key=lambda f: f.start_time)
    return confirmed, held
```

- [ ] **Step 4: get_secondary_provider 추가 (base.py 하단)**

```python
def get_secondary_provider() -> ModelProvider:
    """합의 확정층용 프로바이더 — 검출층과 다른 모델 풀을 쓴다 (독립성 확보)."""
    name = os.getenv("QC_PROVIDER", "gemini")
    if name == "mock":
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise ProviderNotConfiguredError("mock 프로바이더는 자동화 테스트 전용입니다.")
        from app.providers.mock import MockProvider
        return MockProvider()
    from app.providers.gemini import GeminiProvider, DEFAULT_MODEL_POOL
    pool_env = os.getenv("QC_GEMINI_MODELS_SECONDARY")
    if pool_env:
        pool = [m.strip() for m in pool_env.split(",") if m.strip()]
    else:
        # 기본: 검출 풀의 역순 — 검출이 주로 쓰는 모델과 다른 모델이 먼저 온다
        default = os.getenv("QC_GEMINI_MODELS", DEFAULT_MODEL_POOL).split(",")
        pool = [m.strip() for m in reversed(default) if m.strip()]
    if not os.getenv("GEMINI_API_KEY"):
        raise ProviderNotConfiguredError(
            "GEMINI_API_KEY가 설정되지 않았습니다. 검수를 시작할 수 없습니다.")
    return GeminiProvider(model_pool=pool)
```

- [ ] **Step 5: 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_consensus.py -v && venv/bin/python -m pytest -q`
Expected: 전부 passed

- [ ] **Step 6: 커밋**

```bash
git add backend/app/core/consensus.py backend/app/providers/base.py backend/tests/test_consensus.py
git commit -m "feat: 교차 합의 필터 — 블라인드 재청취 + 의미 수준 일치 판정"
```

---

### Task 7: 파이프라인 v3 배선 (STT 격하 + 합의 단계 + 커버리지)

**Files:**
- Modify: `backend/app/core/pipeline.py`
- Modify: `backend/app/core/ingest.py` 호출 방식 (파일 수정 없음 — 호출부만)
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 5 `run_panel` (tuple 반환), Task 6 `confirm_findings`/`get_secondary_provider`
- Produces: `QCPipeline.run(job, on_progress) -> QCResult` — `findings`는 룰 지적 + 합의 확정 지적, `held`는 검출·확정 단계 보류의 합집합(확정 지적이 있는 세그먼트는 제외)
- STT 정책: `kr_srt_path` 있으면 SRT 파싱(표시용), 없으면 env `QC_DISPLAY_STT=1`일 때만 STT 실행. 그 외에는 한국어 텍스트 없이 진행(korean=[]).

- [ ] **Step 1: 실패하는 테스트 추가**

`backend/tests/test_pipeline.py`에 추가 (기존 테스트의 mock 사용 패턴을 따른다):

```python
import pytest

from app.core.pipeline import QCPipeline
from app.providers.mock import MockProvider
from app.schemas import QCJobInput


@pytest.mark.anyio
async def test_pipeline_v3_without_korean_text(tmp_path, monkeypatch):
    # 한국어 SRT도 오디오도 없이 영어 SRT만으로 동작해야 한다 (STT 강제 없음)
    en_srt = tmp_path / "en.srt"
    en_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nMy kidney hurts\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nFine line\n\n",
        encoding="utf-8")
    job = QCJobInput(en_srt_path=str(en_srt))
    result = await QCPipeline(provider=MockProvider()).run(job)
    # korean이 없으면 mock 패턴(korean 필요)이 안 걸려 페르소나 지적 0건이어도,
    # 파이프라인이 예외 없이 완주하고 held가 리스트여야 한다
    assert isinstance(result.held, list)
    assert result.verdict.status in ("pass", "conditional", "fail")


@pytest.mark.anyio
async def test_pipeline_v3_stt_disabled_by_default(tmp_path, monkeypatch):
    called = {"stt": False}

    async def fake_load(lang, srt_path, audio_path):
        if lang == "ko" and audio_path:
            called["stt"] = True
        if lang == "en":
            from app.core.ingest import parse_srt
            return parse_srt(open(srt_path, encoding="utf-8").read())
        return []

    monkeypatch.setattr("app.core.pipeline.load_text_source", fake_load)
    en_srt = tmp_path / "en.srt"
    en_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n\n", encoding="utf-8")
    job = QCJobInput(en_srt_path=str(en_srt), kr_audio_path="/fake/kr.wav")
    monkeypatch.delenv("QC_DISPLAY_STT", raising=False)
    await QCPipeline(provider=MockProvider()).run(job)
    assert called["stt"] is False  # 기본값: STT는 판단 경로에서 제거 — 호출 자체가 없어야 함
```

참고: 두 번째 테스트에서 `kr_audio_path="/fake/kr.wav"`는 존재하지 않는 경로라 씬 클립 추출이 실패하지만, run_panel/confirm_findings는 실패를 로그로 넘기고 계속 진행하도록 되어 있다(우아한 저하).

- [ ] **Step 2: 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — STT가 여전히 호출되거나 held 미존재

- [ ] **Step 3: pipeline.py 교체**

`QCPipeline.run`을 다음으로 교체한다 (룰 체크·억양 체크 블록은 기존 그대로 유지):

```python
    async def run(self, job: QCJobInput, on_progress: Optional[ProgressFn] = None) -> QCResult:
        provider = self.provider or get_provider()
        notify = on_progress or (lambda stage, d, t: None)

        # ① 텍스트 수집 — 영어 SRT 필수. 한국어 텍스트는 표시용일 뿐이다:
        #    SRT가 있으면 파싱, 없으면 QC_DISPLAY_STT=1일 때만 STT (기본은 비활성 —
        #    한국어는 페르소나가 오디오로 직접 듣는다. 설계 스펙 §2 결정 1)
        notify("ingest", 0, 2)
        korean = []
        if job.kr_srt_path:
            korean = await load_text_source("ko", job.kr_srt_path, None)
        elif job.kr_audio_path and os.getenv("QC_DISPLAY_STT") == "1":
            korean = await load_text_source("ko", None, job.kr_audio_path)
        notify("ingest", 1, 2)
        dubbed = await load_text_source("en", job.en_srt_path, None)
        notify("ingest", 2, 2)

        # ② 정렬 + 씬 배정 (korean이 비어도 영어 SRT만으로 pairs가 만들어진다)
        notify("align", 0, 1)
        pairs = assign_scenes(align(korean, dubbed))
        notify("align", 1, 1)

        # ③ 결정론적 룰 체크 — 합의 필터를 거치지 않고 그대로 확정된다
        notify("rules", 0, 1)
        rule_findings = run_text_checks(pairs) + check_sensitive_words(pairs)
        if job.stem_audio_path:
            rule_findings += check_audio_quality(job.stem_audio_path, pairs)
            if job.kr_audio_path:
                try:
                    rule_findings += await asyncio.to_thread(
                        check_dialogue_timing_sync, pairs, job.kr_audio_path, job.stem_audio_path
                    )
                except Exception as e:
                    print(f"[파이프라인] 발화 타이밍 동기화 체크 실패, 해당 체크 없이 진행: {e}")
            try:
                rule_findings += await asyncio.to_thread(
                    check_accent_conformance, pairs, job.stem_audio_path
                )
            except Exception as e:
                print(f"[파이프라인] 억양 분류 실패, 해당 체크 없이 진행: {e}")
        notify("rules", 1, 1)

        # ④ 페르소나 패널 (씬 오디오 직접 청취) → 후보 + 청취 불가 보류
        scenes = group_by_scene(pairs)
        candidates, held_detect = await run_panel(
            scenes, load_knowledge(), provider,
            stem_wav_path=job.stem_audio_path,
            kr_audio_path=job.kr_audio_path,
            on_progress=lambda d, t: notify("panel", d, t),
        )

        # ⑤ 교차 합의 확정 — 페르소나 지적만 대상 (룰 지적은 결정론적이라 면제)
        notify("consensus", 0, 1)
        secondary = self.provider or get_secondary_provider()
        confirmed, held_consensus = await confirm_findings(
            candidates, scenes, job.kr_audio_path, secondary, load_knowledge())
        notify("consensus", 1, 1)

        findings = rule_findings + confirmed
        confirmed_segments = {f.segment_id for f in confirmed}
        held = [h for h in (held_detect + held_consensus)
                if h.segment_id not in confirmed_segments]
        # 같은 세그먼트가 두 단계에서 모두 보류되면 1건만 남긴다
        seen: set = set()
        held = [h for h in held if not (h.segment_id in seen or seen.add(h.segment_id))]

        # ⑥ 판정
        notify("verdict", 0, 1)
        config = load_config()
        axis_scores = compute_axis_scores(findings, n_pairs=len(pairs), config=config)
        verdict = decide(axis_scores, findings, config)
        if held:
            verdict = verdict.model_copy(update={
                "reasons": verdict.reasons + [f"판단 보류 {len(held)}건 (청취 불가 구간)"]})
        notify("verdict", 1, 1)

        return QCResult(verdict=verdict, findings=findings, pairs=pairs, held=held)
```

import 추가:

```python
import os
from app.core.consensus import confirm_findings
from app.providers.base import ModelProvider, get_provider, get_secondary_provider
```

참고: `self.provider`(테스트 주입)가 있으면 확정층도 같은 주입 프로바이더를 쓴다 — 테스트 결정성을 위해서다. 운영 경로(`self.provider is None`)에서만 `get_secondary_provider()`가 다른 모델 풀을 제공한다.

- [ ] **Step 4: 통과 확인**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전부 passed (기존 test_pipeline 테스트 중 STT 폴백을 단언하는 테스트가 있으면 `QC_DISPLAY_STT=1`을 monkeypatch로 설정하도록 수정한다)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: 파이프라인 v3 — STT 기본 비활성, 합의 확정 단계, 판단 보류 커버리지"
```

---

### Task 8: 프론트엔드 — 들은 한국어 병기 · 합의 배지 · 보류 섹션

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/views/ReportView.jsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: API 응답의 `finding.heard_korean`, `finding.consensus`, `result.held[]`

- [ ] **Step 1: App.jsx 상태 추가**

`const [findings, setFindings] = useState([]);` 아래에 추가:

```jsx
  const [held, setHeld] = useState([]); // 판단 보류 구간 (청취 불가 등)
```

`setFindings(result.findings);` 근처(파이프라인 결과 반영부)에 추가:

```jsx
    setHeld(result.held || []);
```

- [ ] **Step 2: 지적 카드에 들은 한국어 + 합의 배지**

App.jsx의 finding 카드 렌더링부(원문/번역 표시 근처)에서 원문 표시 옆에 추가:

```jsx
  {f.heard_korean && (
    <div className="heard-korean" title="모델이 실제로 들은 한국어 — 청취가 정확한지 직접 확인하세요">
      🎧 들은 한국어: {f.heard_korean}
    </div>
  )}
  {f.consensus && <span className="consensus-badge">교차 합의 {f.consensus}</span>}
```

- [ ] **Step 3: 보류 섹션 렌더링**

findings 목록 아래(같은 패널 안)에 추가:

```jsx
  {held.length > 0 && (
    <section className="held-section">
      <h3>판단 보류 구간 ({held.length})</h3>
      <p className="held-hint">청취가 불확실해 검증하지 못한 구간입니다 — 필요하면 직접 들어 확인하세요.</p>
      {held.map((h) => (
        <div key={`${h.scene_id}-${h.segment_id}`} className="held-item"
             onClick={() => { if (videoRef.current) { videoRef.current.currentTime = h.start; videoRef.current.play().catch(() => {}); } }}>
          <span className="held-time">{Math.floor(h.start / 60)}:{String(Math.floor(h.start % 60)).padStart(2, "0")}</span>
          <span>{h.segment_id}</span>
          <span className="held-reason">{h.reason}</span>
        </div>
      ))}
    </section>
  )}
```

- [ ] **Step 4: App.css 스타일 추가**

```css
.heard-korean { font-size: 0.85em; color: var(--text-secondary, #9aa); margin-top: 4px; }
.consensus-badge { font-size: 0.72em; padding: 2px 8px; border-radius: 10px;
  background: rgba(80, 200, 140, 0.15); color: #5fc98f; margin-left: 8px; }
.held-section { margin-top: 18px; padding: 12px; border: 1px dashed var(--border, #445);
  border-radius: 8px; }
.held-hint { font-size: 0.8em; opacity: 0.7; }
.held-item { display: flex; gap: 12px; padding: 6px 4px; cursor: pointer; font-size: 0.85em; }
.held-item:hover { background: rgba(255, 255, 255, 0.05); }
.held-time { font-family: monospace; opacity: 0.8; }
.held-reason { margin-left: auto; color: #d9a04a; }
```

참고: 기존 App.css의 변수명(`--text-secondary`, `--border`)이 다르면 그 파일에서 실제 쓰는 변수명으로 맞춘다.

- [ ] **Step 5: 수동 검증**

```bash
QC_PROVIDER=mock는 운영 차단이므로 실제 검증은 mock 테스트로 대신하고, UI는 다음으로 확인:
cd frontend && npm run dev
```

브라우저에서 렌더 오류 없는지, 기존 리포트 화면이 깨지지 않는지 확인. (held/heard_korean 없는 기존 결과에서도 조건부 렌더라 안전해야 함)

- [ ] **Step 6: 커밋**

```bash
git add frontend/src/App.jsx frontend/src/views/ReportView.jsx frontend/src/App.css
git commit -m "feat: 검수 화면 — 들은 한국어 병기, 교차 합의 배지, 판단 보류 섹션"
```

---

### Task 9: 통합 테스트 + 문서 갱신

**Files:**
- Test: `backend/tests/test_pipeline.py` (통합 테스트 추가)
- Modify: `PROJECT_OVERVIEW.md` (v3 구현 상태 반영)

- [ ] **Step 1: mock 전체 경로 통합 테스트 추가**

```python
@pytest.mark.anyio
async def test_pipeline_v3_end_to_end_with_kr_srt(tmp_path):
    # 한국어 SRT가 있는 구성: mock 패턴이 걸려 지적이 나오고, 합의(mock은 항상 동의)로 확정된다
    en_srt = tmp_path / "en.srt"
    en_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nMy kidney hurts\n\n", encoding="utf-8")
    kr_srt = tmp_path / "kr.srt"
    kr_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n어이가 없네\n\n", encoding="utf-8")
    job = QCJobInput(en_srt_path=str(en_srt), kr_srt_path=str(kr_srt))
    result = await QCPipeline(provider=MockProvider()).run(job)
    persona_findings = [f for f in result.findings if f.source.startswith("persona:")]
    assert persona_findings, "mock kidney 패턴이 지적으로 나와야 한다"
    # kr_audio_path가 없으므로 합의는 통과(pass-through) — consensus 미표시가 정상
    assert all(f.consensus == "" for f in persona_findings)
```

- [ ] **Step 2: 전체 테스트**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전부 passed

- [ ] **Step 3: PROJECT_OVERVIEW.md 상단 구현 상태 블록 갱신**

`> **구현 상태**:` 인용 블록을 다음으로 교체:

```markdown
> **구현 상태**: 본문(5축 MOS)은 v1 기준 설명이며, 현재 코드는 **v3(오디오 직접
> 청취 + 교차 합의)**가 구현되어 있습니다. v3의 설계 근거와 벤치마크는
> [`2026-07-20-audio-direct-consensus-design.md`](docs/superpowers/specs/2026-07-20-audio-direct-consensus-design.md)
> 참고. 7축 확장(v2) 설계는 [`2026-07-15-dubbing-qc-v2-audio-redesign.md`](docs/superpowers/specs/2026-07-15-dubbing-qc-v2-audio-redesign.md).
```

- [ ] **Step 4: 커밋**

```bash
git add backend/tests/test_pipeline.py PROJECT_OVERVIEW.md
git commit -m "test: v3 통합 테스트 + 문서 구현 상태 갱신"
```

---

## Self-Review 결과

- **스펙 커버리지**: 결정 1(STT 우회) → Task 1·5·7 / 결정 2(합의) → Task 6 / 결정 3(씬당 1요청·맥락·heard_korean·429) → Task 4·5 / 커버리지 표시 → Task 2·7·8 / 무료 저속 모드(모델 풀) → Task 4 / 프론트 → Task 8. 스펙 §7 보류 항목(GPT 교차, 운영 모델 재검증)은 의도적으로 구현 스코프 밖.
- **플레이스홀더**: 없음 — 모든 코드 블록은 그대로 붙여넣어 동작하는 수준으로 작성.
- **타입 일관성**: `JudgeOutput`(Task 2 정의 → 3·4·5·6 사용), `HeldSegment`(2 → 5·6·7·8), `extract_scene_clip(audio_path, start, end, ctx_seconds, pad, out_dir)`(1 → 5·6), `complete_json`(3 → 4·6), `run_panel` tuple 반환(5 → 7) 확인 완료.
