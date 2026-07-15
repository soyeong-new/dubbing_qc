# 더빙 QC v2 오디오/음성 확장 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v1(텍스트 로컬라이제이션 QC)에 오디오/음성 차원(억양 적합성, 민감어·욕설,
감정 표현 원본 대조)을 더해 MOS를 5축→6축으로 확장하고, 씬 배치에 크기 상한
안전장치를 추가한다.

**Architecture:** 기존 파이프라인(ingest→align→rule_checks→judge_panel→verdict)을
그 자리에서 확장한다. 새 무거운 의존성은 억양 분류 모델(SpeechBrain) 하나뿐이고,
민감어 체크는 사전 1차 필터(결정론적) + 원어민 페르소나의 기존 호출에 통합된
LLM 2차 판단으로 구성된다(별도 API 호출 추가 없음).

**Tech Stack:** 기존 스택(Python/FastAPI/Pydantic v2/pytest, React/Vite) +
SpeechBrain/torch/torchaudio(억양 분류 전용, 신규)

## Global Constraints

- Gemini 모델 ID는 정확히 `'gemini-3.5-flash'`.
- mock 프로바이더는 `PYTEST_CURRENT_TEST` 환경변수가 있을 때만 선택 가능 —
  이 게이트를 절대 우회하지 않는다 (v1 §4 원칙 그대로).
- `AXES`(`backend/app/schemas.py`)는 6개 문자열이 되어야 한다: 정확히
  `["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성", "억양 적합성"]`.
  **"민감어·욕설"은 AXES에 들어가지 않는다** — MOS 점수화 대상이 아니라 별도
  콘텐츠 플래그이기 때문이다 (설계 스펙의 "7축" 표현은 부정확한 요약이었음 —
  실제로는 MOS 6축 + 콘텐츠 플래그 1종).
- `QCFinding.finding_type`은 정확히 `Literal["quality", "sensitive"]`, 기본값
  `"quality"`. `finding_type == "sensitive"`인 finding은 `verdict.py`의
  `compute_axis_scores`에서 반드시 제외되어야 한다 (axis 라벨이 우연히 일치해도).
- `QCFinding.description`은 반드시 한국어, `recommendation`은 반드시 영어
  (기존 v1 관례 그대로).
- 싱크 정확도는 **이미 v1에 구현되어 있다** (`rule_checks.py`의
  `check_sync_overflow`, axis="싱크 정확도") — 이 계획에서 재구현하지 않는다.
- 새 런타임 의존성: `torch`, `torchaudio`, `speechbrain` (억양 분류 전용).
  이 모델들은 무겁고(다운로드 수백 MB~GB) 자동화 테스트에서 실제로 로드하지
  않는다 — 모든 테스트는 주입 가능한 함수(`classify_fn`)로 실제 모델을 대체한다.
- 백엔드 테스트는 `backend/` 디렉토리에서 `venv/bin/python -m pytest`로 실행.

---

### Task 1: 스키마 확장 — AXES 6축 + finding_type 필드

**Files:**
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/core/verdict.py`
- Test: `backend/tests/test_schemas.py`
- Test: `backend/tests/test_verdict.py`

**Interfaces:**
- Produces: `AXES: list[str]` (6개, 억양 적합성 추가), `QCFinding.finding_type: Literal["quality", "sensitive"] = "quality"`
- Consumes: 없음 (기반 태스크)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_schemas.py`의 기존 파일 끝에 추가:

```python
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
```

`backend/tests/test_verdict.py`의 기존 파일 끝에 추가:

```python
def test_sensitive_findings_excluded_from_axis_scoring():
    config = load_config()
    findings = [
        finding("언어 적합성", "high", seg="p1"),
    ]
    findings[0] = findings[0].model_copy(update={"finding_type": "sensitive"})
    scores = compute_axis_scores(findings, n_pairs=100, config=config)
    lang = next(s for s in scores if s.axis == "언어 적합성")
    assert lang.mos == 5  # 민감어 finding은 감점에 반영되지 않아야 한다
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_schemas.py tests/test_verdict.py -v`
Expected: FAIL — `AXES`가 5개뿐이라 `test_axes_has_six_entries_including_accent` 실패,
`finding_type` 속성이 없어 `AttributeError`

- [ ] **Step 3: schemas.py 수정**

`backend/app/schemas.py`에서 다음 줄:
```python
AXES = ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성"]
```
을 다음으로 교체:
```python
AXES = ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성", "억양 적합성"]
```

`QCFinding` 클래스의 `alternatives` 필드 다음 줄에 추가:
```python
    finding_type: Literal["quality", "sensitive"] = "quality"
```

- [ ] **Step 4: verdict.py 수정**

`backend/app/core/verdict.py`의 `compute_axis_scores` 함수에서:
```python
        total = sum(deduction_w.get(f.severity, 0) for f in findings if f.axis == axis)
```
를 다음으로 교체:
```python
        total = sum(
            deduction_w.get(f.severity, 0) for f in findings
            if f.axis == axis and f.finding_type == "quality"
        )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_schemas.py tests/test_verdict.py -v`
Expected: 전체 PASS (기존 `test_all_axes_always_scored`도 `AXES`를 그대로
참조하므로 자동으로 6개 축 기준으로 통과)

- [ ] **Step 6: 전체 회귀 확인 및 커밋**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

```bash
git add backend/app/schemas.py backend/app/core/verdict.py backend/tests/test_schemas.py backend/tests/test_verdict.py
git commit -m "feat: MOS AXES를 6축으로 확장(억양 적합성 추가), QCFinding에 finding_type 필드 추가"
```

---

### Task 2: 씬(배치) 배정 크기 상한 안전장치

**Files:**
- Modify: `backend/app/core/alignment.py`
- Test: `backend/tests/test_alignment.py`

**Interfaces:**
- Consumes: Task 1 없음 (독립)
- Produces: `assign_scenes(pairs, gap_threshold=3.0, max_segments=20, max_duration=180.0) -> list[AlignedPair]` — 시그니처 확장 (기존 호출부는 기본값으로 동작, 하위 호환)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_alignment.py` 파일 끝에 추가 (기존 `kr`/`en` 헬퍼 함수 재사용):

```python
def test_assign_scenes_size_cap_splits_long_uninterrupted_dialogue():
    # 간격 없이(각 세그먼트 gap_threshold 미만) 25개 세그먼트가 이어짐 → 20개 상한 초과
    pairs = []
    t = 0.0
    for i in range(25):
        pairs.append(kr(t, t + 1.0, f"line{i}"))
        t += 1.5  # 세그먼트 간 간격 0.5초 (gap_threshold 3.0초 미만)
    dubbed = [en(p.korean.start, p.korean.end, p.korean.text) for p in
              align(pairs, [en(pp.start, pp.end, pp.text) for pp in pairs])]
    aligned = align(pairs, dubbed)
    result = assign_scenes(aligned, max_segments=20, max_duration=999.0)
    scene_ids = [p.scene_id for p in result]
    assert scene_ids[0] == "scene_1"
    assert scene_ids[20] == "scene_2"  # 21번째 세그먼트(인덱스20)부터 새 씬


def test_assign_scenes_size_cap_does_not_trigger_on_short_dialogue():
    pairs = align([kr(0, 1, "a"), kr(1.5, 2.5, "b")], [en(0, 1, "a"), en(1.5, 2.5, "b")])
    result = assign_scenes(pairs, max_segments=20, max_duration=180.0)
    assert result[0].scene_id == result[1].scene_id  # 상한 안 걸리면 씬 안 나뉨
```

주의: 위 테스트는 `align()`/`kr()`/`en()` 기존 헬퍼를 사용하는데, 실제로는
간단히 하기 위해 `kr`만으로 `AlignedPair`를 직접 만드는 편이 낫다. 아래처럼
`test_alignment.py`에 이미 있는 `kr`/`en` 헬퍼 시그니처(`kr(s, e, t)`,
`en(s, e, t)`가 `SegmentText` 반환)를 그대로 재사용하되, 정렬을 거치지 않고
`AlignedPair`를 직접 구성하도록 아래 최종본으로 작성한다:

```python
def test_assign_scenes_size_cap_splits_long_uninterrupted_dialogue():
    from app.schemas import AlignedPair
    pairs = []
    t = 0.0
    for i in range(25):
        seg = kr(t, t + 1.0, f"line{i}")
        pairs.append(AlignedPair(id=f"pair_{i+1}", korean=seg, dubbed=seg, alignment_confidence=1.0))
        t += 1.5  # 세그먼트 간 간격 0.5초 (gap_threshold 3.0초 미만)
    result = assign_scenes(pairs, max_segments=20, max_duration=999.0)
    scene_ids = [p.scene_id for p in result]
    assert scene_ids[0] == "scene_1"
    assert scene_ids[20] == "scene_2"


def test_assign_scenes_size_cap_does_not_trigger_on_short_dialogue():
    from app.schemas import AlignedPair
    seg_a, seg_b = kr(0, 1, "a"), kr(1.5, 2.5, "b")
    pairs = [
        AlignedPair(id="pair_1", korean=seg_a, dubbed=seg_a, alignment_confidence=1.0),
        AlignedPair(id="pair_2", korean=seg_b, dubbed=seg_b, alignment_confidence=1.0),
    ]
    result = assign_scenes(pairs, max_segments=20, max_duration=180.0)
    assert result[0].scene_id == result[1].scene_id
```

(위 두 블록 중 **두 번째(최종본)만 실제 파일에 반영**한다 — 첫 블록은 왜 이
형태로 정착했는지 보여주기 위한 설명용이며 파일에 넣지 않는다.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: FAIL — `assign_scenes()`가 `max_segments`/`max_duration` 인자를
받지 않아 `TypeError`

- [ ] **Step 3: alignment.py 수정**

`backend/app/core/alignment.py`의 `assign_scenes` 함수 전체를 다음으로 교체:

```python
def assign_scenes(pairs: List[AlignedPair], gap_threshold: float = 3.0,
                  max_segments: int = 20, max_duration: float = 180.0) -> List[AlignedPair]:
    scene_num = 1
    prev_end = None
    scene_start = None
    scene_count = 0
    for p in pairs:
        anchor = p.korean or p.dubbed
        new_scene = False
        if prev_end is not None and anchor.start - prev_end > gap_threshold:
            new_scene = True
        elif scene_start is not None and (
            scene_count >= max_segments or anchor.end - scene_start > max_duration
        ):
            # 크기 상한(안전장치) — 침묵 기준만으로 배치가 과도하게 커지는
            # 드문 경우에만 개입한다. 대다수 배치는 이 조건에 도달하지 않는다.
            new_scene = True
        if new_scene:
            scene_num += 1
            scene_start = None
            scene_count = 0
        if scene_start is None:
            scene_start = anchor.start
        scene_count += 1
        p.scene_id = f"scene_{scene_num}"
        prev_end = max(prev_end or 0.0, anchor.end)
    return pairs
```

- [ ] **Step 4: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS (기존 `test_assign_scenes_by_gap` 등 회귀 없음 — 새 파라미터는
기본값이 넉넉해 기존 테스트 케이스에서 발동하지 않음)

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/alignment.py backend/tests/test_alignment.py
git commit -m "feat: 씬 배치에 크기 상한 안전장치 추가 (침묵 기준 유지, 드문 예외만 개입)"
```

---

### Task 3: 민감어 사전 + 결정론적 1차 필터

**Files:**
- Create: `backend/app/knowledge/sensitive_words.yaml`
- Modify: `backend/app/core/rule_checks.py`
- Test: `backend/tests/test_rule_checks.py`

**Interfaces:**
- Consumes: Task 1의 `QCFinding.finding_type`
- Produces: `load_sensitive_terms(path=None) -> list[tuple[str, str]]`,
  `check_sensitive_words(pairs, terms=None) -> list[QCFinding]` (각 finding은
  `finding_type="sensitive"`, `axis="언어 적합성"`)

- [ ] **Step 1: 사전 파일 작성**

`backend/app/knowledge/sensitive_words.yaml`:

```yaml
# 민감어·욕설 사전 — 1차 결정론적 필터 (명백한 경우만 포착)
# 실제 운영 목록은 사내 심의 기준 문서를 확보한 뒤 채워야 한다
# (설계 스펙 docs/superpowers/specs/2026-07-15-dubbing-qc-v2-audio-redesign.md §11 참조).
# 아래는 메커니즘 검증용 자리표시 항목이다 — 실제 욕설/슬러를 넣지 않는다.
terms:
  - word: "placeholder-slur-1"
    category: "인종/민족"
  - word: "placeholder-profanity-1"
    category: "욕설"
```

- [ ] **Step 2: 실패하는 테스트 작성**

`backend/tests/test_rule_checks.py` 파일 끝에 추가:

```python
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
```

(`pair(...)` 헬퍼는 `test_rule_checks.py`에 이미 정의되어 있다 — 재사용한다.)

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_rule_checks.py -v`
Expected: FAIL — `load_sensitive_terms`/`check_sensitive_words`가 아직 없어
`ImportError`

- [ ] **Step 4: rule_checks.py에 추가**

`backend/app/core/rule_checks.py` 상단 import 블록에 추가:
```python
from pathlib import Path
import yaml
```

파일 끝에 추가:
```python
_DEFAULT_SENSITIVE_WORDS = Path(__file__).parent.parent / "knowledge" / "sensitive_words.yaml"


def load_sensitive_terms(path: str = None) -> List[tuple]:
    p = Path(path) if path else _DEFAULT_SENSITIVE_WORDS
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [(t["word"].lower(), t.get("category", "기타")) for t in data.get("terms", [])]


def check_sensitive_words(pairs: List[AlignedPair], terms: List[tuple] = None) -> List[QCFinding]:
    terms = terms if terms is not None else load_sensitive_terms()
    findings = []
    for p in pairs:
        if not p.dubbed or not p.dubbed.text.strip():
            continue
        text_lower = p.dubbed.text.lower()
        for word, category in terms:
            if word in text_lower:
                anchor = p.korean or p.dubbed
                findings.append(QCFinding(
                    id=f"rule_sensitive_{p.id}_{word.replace(' ', '_')}",
                    segment_id=p.id, category="localization", severity="high",
                    issue_type=f"민감어({category})",
                    start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
                    description=f"금칙어 사전에 등록된 표현이 감지되었습니다 (분류: {category}). "
                                "해당 표현의 사용 맥락과 등급 영향을 검토하세요.",
                    original_text=p.korean.text if p.korean else "",
                    current_translation=p.dubbed.text,
                    recommendation="해당 표현을 검토하고 필요 시 수정하세요.",
                    confidence=1.0, axis="언어 적합성", source="rule",
                    finding_type="sensitive",
                ))
                break
    return findings
```

- [ ] **Step 5: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_rule_checks.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add backend/app/knowledge/sensitive_words.yaml backend/app/core/rule_checks.py backend/tests/test_rule_checks.py
git commit -m "feat: 민감어 사전 1차 결정론적 필터 추가 (finding_type=sensitive)"
```

---

### Task 4: 원어민 페르소나에 민감어 판단 통합 (finding_type 파싱)

**Files:**
- Modify: `backend/app/core/judge_panel.py`
- Modify: `backend/app/providers/gemini.py`
- Test: `backend/tests/test_judge_panel.py`
- Test: `backend/tests/test_gemini_provider.py`

**Interfaces:**
- Consumes: Task 1의 `QCFinding.finding_type`
- Produces: `parse_judge_response`가 응답의 `finding_type` 필드를 파싱해
  `QCFinding.finding_type`에 반영 (없거나 잘못된 값이면 `"quality"`로 폴백)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_gemini_provider.py` 파일 끝에 추가:

```python
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
```

`backend/tests/test_judge_panel.py` 파일 끝에 추가:

```python
def test_native_persona_instruction_mentions_sensitive_content():
    native = next(p for p in PERSONAS if p.key == "native")
    assert "민감" in native.instruction
    assert "finding_type" in native.instruction
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py tests/test_judge_panel.py -v`
Expected: FAIL — `finding_type` 파싱 미구현, 원어민 지시문에 민감어 언급 없음

- [ ] **Step 3: gemini.py 수정**

`backend/app/providers/gemini.py`의 `JUDGE_PROMPT_TEMPLATE`에서 다음 줄:
```python
- axis는 다음 중 하나: {axes}
- severity는 "high" | "medium" | "low"
```
을 다음으로 교체:
```python
- axis는 다음 중 하나: {axes}
- severity는 "high" | "medium" | "low"
- finding_type은 "quality"(일반 품질 지적) 또는 "sensitive"(민감어·욕설 지적) 중 하나.
  생략 시 "quality"로 간주됩니다.
```

같은 파일의 반환 스키마 예시 줄:
```python
반환 스키마:
[{{"segment_id": "...", "severity": "...", "issue_type": "...",
  "description": "...", "recommendation": "...", "confidence": 0.9, "axis": "..."}}]
```
을 다음으로 교체:
```python
반환 스키마:
[{{"segment_id": "...", "severity": "...", "issue_type": "...",
  "description": "...", "recommendation": "...", "confidence": 0.9, "axis": "...",
  "finding_type": "quality"}}]
```

`parse_judge_response` 함수에서 `QCFinding(...)` 생성 부분 바로 위에 추가:
```python
        finding_type = item.get("finding_type", "quality")
        if finding_type not in ("quality", "sensitive"):
            finding_type = "quality"
```
그리고 `QCFinding(...)` 호출에 인자 추가:
```python
            source=f"persona:{persona.key}",
            finding_type=finding_type,
        ))
```
(기존 `source=f"persona:{persona.key}",\n        ))`의 닫는 괄호 바로 앞에 삽입)

- [ ] **Step 4: judge_panel.py의 원어민 페르소나 지시문 수정**

`backend/app/core/judge_panel.py`의 `native` Persona 정의를 다음으로 교체:

```python
    Persona(
        key="native", name="영어 원어민 시청자",
        axes=["자연스러움", "언어 적합성"],
        instruction=(
            "당신은 한국어를 전혀 모르는 미국인 시청자입니다. korean 필드는 무시하고 "
            "english_dub만 읽으십시오. 확인할 것: (1) 원어민이 실제로 쓰는 표현인가, "
            "번역투인가 (2) 구어체 대사로서 리듬이 자연스러운가 (3) 어색하거나 "
            "우스꽝스럽게 들리는 문장이 있는가. 의미의 정확성은 평가하지 마십시오. "
            "추가로, 사전 필터에 걸리지 않은 애매한 민감 표현도 함께 확인하십시오: "
            "(4) 인종/성/종교/정치적으로 민감하거나 암시적으로 차별적인 표현이 있는가 "
            "(5) 등급(심의)에 영향을 줄 수 있는 수위의 욕설·폭력적 표현이 있는가. "
            "(1)~(3)에 해당하는 지적은 finding_type을 \"quality\"로, (4)~(5)에 "
            "해당하는 지적은 finding_type을 \"sensitive\"로 표기하십시오."
        ),
    ),
```

- [ ] **Step 5: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py tests/test_judge_panel.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add backend/app/providers/gemini.py backend/app/core/judge_panel.py backend/tests/test_gemini_provider.py backend/tests/test_judge_panel.py
git commit -m "feat: 원어민 페르소나에 민감어 판단 통합, finding_type 파싱 추가 (별도 LLM 패스 없음)"
```

---

### Task 5: 연출가 페르소나에 원본 오디오 입력 추가

**Files:**
- Modify: `backend/app/providers/base.py`
- Modify: `backend/app/providers/gemini.py`
- Modify: `backend/app/providers/mock.py`
- Modify: `backend/app/core/judge_panel.py`
- Test: `backend/tests/test_gemini_provider.py`
- Test: `backend/tests/test_judge_panel.py`

**Interfaces:**
- Consumes: Task 4의 `parse_judge_response` (변경 없음, 그대로 사용)
- Produces: `ModelProvider.judge(..., audio_clip_path=None, original_audio_clip_path=None)`,
  `run_panel(scenes, knowledge, provider, stem_wav_path=None, kr_audio_path=None, on_progress=None)`

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_gemini_provider.py` 파일 끝에 추가:

```python
def test_judge_attaches_both_original_and_dub_audio(monkeypatch, tmp_path):
    import asyncio
    from unittest.mock import MagicMock
    from app.providers.gemini import GeminiProvider

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
        [PAIR], PERSONA, knowledge="",
        audio_clip_path=str(dub_wav), original_audio_clip_path=str(orig_wav),
    ))
    # 오디오 파트가 2개(원본+더빙) 포함되어야 한다
    audio_parts = [p for p in captured["parts"] if isinstance(p, dict) and "mime_type" in p]
    assert len(audio_parts) == 2
```

`backend/tests/test_judge_panel.py` 파일 끝에 추가:

```python
async def test_run_panel_extracts_original_audio_clip_for_director(monkeypatch, tmp_path):
    import wave, struct
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()

    def make_wav(path, seconds=3.0, rate=16000):
        n = int(seconds * rate)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(struct.pack(f"{n}h", *([0] * n)))

    stem_path = tmp_path / "stem.wav"
    kr_path = tmp_path / "kr.wav"
    make_wav(stem_path)
    make_wav(kr_path)

    pair = AlignedPair(
        id="pair_1", scene_id="scene_1",
        korean=SegmentText(start=0, end=2, speaker="A", text="어이가 없네."),
        dubbed=SegmentText(start=0, end=2, speaker="A", text="I have no kidney."),
    )
    received = {"original_audio_clip_path": []}

    class RecordingProvider:
        async def transcribe(self, audio_path, lang):
            return []

        async def judge(self, pairs, persona, knowledge, audio_clip_path=None,
                        original_audio_clip_path=None):
            received["original_audio_clip_path"].append(original_audio_clip_path)
            return await provider.judge(pairs, persona, knowledge, audio_clip_path=audio_clip_path)

    await run_panel(
        {"scene_1": [pair]}, knowledge="", provider=RecordingProvider(),
        stem_wav_path=str(stem_path), kr_audio_path=str(kr_path),
    )
    # director(uses_audio=True) 호출에서만 원본 클립 경로가 채워져야 한다
    director_calls = [v for v in received["original_audio_clip_path"] if v is not None]
    assert len(director_calls) == 1
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py tests/test_judge_panel.py -v`
Expected: FAIL — `judge()`가 `original_audio_clip_path`를 받지 않아 `TypeError`,
`run_panel()`이 `kr_audio_path`를 받지 않아 `TypeError`

- [ ] **Step 3: base.py 수정**

`backend/app/providers/base.py`의 `ModelProvider.judge` 추상 메서드 시그니처를:
```python
    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        ...
```
다음으로 교체:
```python
    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        ...
```

- [ ] **Step 4: gemini.py의 judge() 수정**

`backend/app/providers/gemini.py`의 `GeminiProvider.judge` 메서드 전체를 교체:

```python
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        model = self._genai.GenerativeModel(MODEL_NAME)
        prompt = build_judge_prompt(pairs, persona, knowledge)
        parts = [prompt]
        if audio_clip_path and persona.uses_audio and os.path.exists(audio_clip_path):
            dub_audio = await asyncio.to_thread(_compress_to_mp3, audio_clip_path)
            parts.insert(0, {"mime_type": "audio/mp3", "data": dub_audio})
            parts.insert(0, "[다음 오디오는 영어 더빙입니다]")
        if original_audio_clip_path and persona.uses_audio and os.path.exists(original_audio_clip_path):
            orig_audio = await asyncio.to_thread(_compress_to_mp3, original_audio_clip_path)
            parts.insert(0, {"mime_type": "audio/mp3", "data": orig_audio})
            parts.insert(0, "[다음 오디오는 한국어 원본입니다]")
        # 동기 SDK 호출을 스레드로 넘겨 이벤트 루프가 다른 요청(진행률 폴링 등)을
        # 계속 처리할 수 있게 한다 — transcribe()의 동일 주석 참고.
        response = await asyncio.to_thread(
            model.generate_content,
            parts, generation_config={"response_mime_type": "application/json"},
        )
        return parse_judge_response(response.text, pairs, persona)
```

- [ ] **Step 5: mock.py의 judge() 시그니처 갱신**

`backend/app/providers/mock.py`의 `MockProvider.judge` 시그니처를:
```python
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None) -> List[QCFinding]:
```
다음으로 교체 (본문은 변경 없음, 파라미터만 추가하고 무시):
```python
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
```

- [ ] **Step 6: judge_panel.py의 run_panel() 수정**

`backend/app/core/judge_panel.py`의 `run_panel` 함수 전체를 교체:

```python
async def run_panel(scenes: Dict[str, List[AlignedPair]], knowledge: str,
                    provider: ModelProvider, stem_wav_path: Optional[str] = None,
                    kr_audio_path: Optional[str] = None,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> List[QCFinding]:
    from app.core.rule_checks import extract_clip

    all_findings: List[QCFinding] = []
    scene_ids = sorted(scenes.keys(), key=lambda s: int(s.split("_")[1]))
    for done, scene_id in enumerate(scene_ids, start=1):
        pairs = scenes[scene_id]
        clip_path = None
        original_clip_path = None
        if stem_wav_path:
            anchors = [(p.dubbed or p.korean) for p in pairs if (p.dubbed or p.korean)]
            if anchors:
                try:
                    # extract_clip은 ffmpeg를 동기 호출한다 — asyncio 이벤트 루프를
                    # 막지 않도록 스레드로 넘긴다 (그렇지 않으면 이 씬을 처리하는 동안
                    # 진행률 폴링 등 다른 요청이 전부 멈춘다).
                    clip_path = await asyncio.to_thread(
                        extract_clip, stem_wav_path, anchors[0].start, anchors[-1].end
                    )
                except Exception as e:
                    print(f"[패널] {scene_id} 오디오 클립 추출 실패, 오디오 없이 진행: {e}")
        if kr_audio_path:
            kr_anchors = [p.korean for p in pairs if p.korean]
            if kr_anchors:
                try:
                    original_clip_path = await asyncio.to_thread(
                        extract_clip, kr_audio_path, kr_anchors[0].start, kr_anchors[-1].end
                    )
                except Exception as e:
                    print(f"[패널] {scene_id} 원본 오디오 클립 추출 실패, 원본 없이 진행: {e}")
        for persona in PERSONAS:
            audio = clip_path if persona.uses_audio else None
            orig_audio = original_clip_path if persona.uses_audio else None
            for attempt in (1, 2):
                try:
                    all_findings.extend(
                        await provider.judge(
                            pairs, persona, knowledge,
                            audio_clip_path=audio, original_audio_clip_path=orig_audio,
                        )
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[패널] {scene_id}/{persona.key} 분석 실패 (2회 시도): {e}")
        if on_progress:
            on_progress(done, len(scene_ids))
    return merge_findings(all_findings)
```

- [ ] **Step 7: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_gemini_provider.py tests/test_judge_panel.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

- [ ] **Step 8: 커밋**

```bash
git add backend/app/providers/base.py backend/app/providers/gemini.py backend/app/providers/mock.py backend/app/core/judge_panel.py backend/tests/test_gemini_provider.py backend/tests/test_judge_panel.py
git commit -m "feat: 연출가 페르소나에 원본 한국어 오디오 클립 동시 입력 추가"
```

---

### Task 6: 억양 적합성 분류 모듈 (신규)

**Files:**
- Create: `backend/app/core/accent.py`
- Modify: `backend/requirements.txt`
- Test: `backend/tests/test_accent.py`

**Interfaces:**
- Consumes: Task 1의 `QCFinding.axis` (AXES에 "억양 적합성" 포함), Task 5의 `extract_clip`
- Produces: `classify_accent(wav_path: str) -> tuple[str, float]` (라벨, 확신도),
  `check_accent_conformance(pairs, stem_wav_path, extract_clip_fn=None, classify_fn=None, target_accent="us", confidence_threshold=0.6) -> list[QCFinding]`

**중요:** 실제 SpeechBrain 모델(`classify_accent`/`_get_classifier`)은 무겁고
느린 다운로드가 필요하다. 자동화 테스트는 `classify_fn`을 주입해 실제 모델을
**절대 로드하지 않는다** — `check_srt_audio_match`의 `extract_clip_fn` 주입
패턴과 동일하다.

- [ ] **Step 1: requirements.txt에 의존성 추가**

`backend/requirements.txt` 파일 끝에 추가:
```text
torch>=2.1.0
torchaudio>=2.1.0
speechbrain>=1.0.0
```

- [ ] **Step 2: 실패하는 테스트 작성**

`backend/tests/test_accent.py` (신규 파일):

```python
from app.core.accent import check_accent_conformance
from app.schemas import AlignedPair, SegmentText


def pair(pid="p1", en_text="line"):
    return AlignedPair(
        id=pid,
        korean=SegmentText(start=0.0, end=2.0, speaker="A", text="대사"),
        dubbed=SegmentText(start=0.0, end=2.0, speaker="A", text=en_text),
    )


def fake_extract_clip(src, start, end):
    return src  # 실제 ffmpeg 호출 없이 원본 경로 반환


def test_check_accent_conformance_flags_non_target_accent():
    def fake_classify(clip_path):
        return "british", 0.9

    findings = check_accent_conformance(
        [pair()], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=fake_classify,
    )
    assert len(findings) == 1
    assert findings[0].axis == "억양 적합성"
    assert findings[0].finding_type == "quality"


def test_check_accent_conformance_passes_target_accent():
    def fake_classify(clip_path):
        return "us", 0.95

    findings = check_accent_conformance(
        [pair()], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=fake_classify,
    )
    assert findings == []


def test_check_accent_conformance_flags_low_confidence_even_if_target_label():
    def fake_classify(clip_path):
        return "us", 0.2  # 라벨은 맞지만 확신도가 낮음

    findings = check_accent_conformance(
        [pair()], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=fake_classify,
        confidence_threshold=0.6,
    )
    assert len(findings) == 1


def test_check_accent_conformance_skips_missing_dub_text():
    findings = check_accent_conformance(
        [pair(en_text="")], stem_wav_path="/tmp/stem.wav",
        extract_clip_fn=fake_extract_clip, classify_fn=lambda c: ("us", 0.9),
    )
    assert findings == []
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_accent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.accent'`

- [ ] **Step 4: accent.py 구현**

`backend/app/core/accent.py`:

```python
from typing import Callable, List, Optional
from app.schemas import AlignedPair, QCFinding

MODEL_SOURCE = "Jzuluaga/accent-id-commonaccent_ecapa"  # 실제 모델 ID는 도입 시 재검증
MODEL_SAVEDIR = "/tmp/aether_accent_model"

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    from speechbrain.inference.classifiers import EncoderClassifier
    _classifier = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=MODEL_SAVEDIR)
    return _classifier


def classify_accent(wav_path: str) -> tuple:
    """단일 오디오 클립의 억양을 분류한다. (라벨, 확신도)를 반환한다.

    실제 SpeechBrain 모델을 로드하므로 자동화 테스트에서는 호출하지 않는다 —
    check_accent_conformance()의 classify_fn 주입으로 대체한다.
    """
    classifier = _get_classifier()
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    return text_lab[0], float(score[0])


def check_accent_conformance(
    pairs: List[AlignedPair], stem_wav_path: str,
    extract_clip_fn: Optional[Callable] = None,
    classify_fn: Optional[Callable] = None,
    target_accent: str = "us",
    confidence_threshold: float = 0.6,
) -> List[QCFinding]:
    from app.core.rule_checks import extract_clip as default_extract_clip
    extract_clip_fn = extract_clip_fn or default_extract_clip
    classify_fn = classify_fn or classify_accent

    findings = []
    for p in pairs:
        if not p.dubbed or not p.dubbed.text.strip():
            continue
        clip = extract_clip_fn(stem_wav_path, p.dubbed.start, p.dubbed.end)
        label, confidence = classify_fn(clip)
        if label.lower() != target_accent or confidence < confidence_threshold:
            findings.append(QCFinding(
                id=f"accent_{p.id}", segment_id=p.id, category="voice",
                severity="medium", issue_type="억양 부적합",
                start_time=p.dubbed.start, end_time=p.dubbed.end, speaker=p.dubbed.speaker,
                description=f"이 세그먼트의 억양이 목표 표준과 다르게 분류되었습니다 "
                            f"(분류: {label}, 확신도 {confidence:.2f}).",
                original_text=p.korean.text if p.korean else "",
                current_translation=p.dubbed.text,
                recommendation="목표 억양으로 재녹음하거나 성우 캐스팅을 검토하세요.",
                confidence=confidence, axis="억양 적합성", source="rule",
            ))
    return findings
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_accent.py -v`
Expected: 전체 PASS (실제 torch/speechbrain 다운로드 없이 통과 — `classify_fn`
주입 덕분)

- [ ] **Step 6: 의존성 설치 및 전체 회귀 확인**

Run: `cd backend && venv/bin/pip install -r requirements.txt`
(참고: torch/speechbrain 설치는 수백 MB~1GB 다운로드가 발생할 수 있다)

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

- [ ] **Step 7: 커밋**

```bash
git add backend/app/core/accent.py backend/requirements.txt backend/tests/test_accent.py
git commit -m "feat: 억양 적합성 분류 모듈 추가 (SpeechBrain, 세그먼트 독립 판단)"
```

---

### Task 7: 파이프라인 오케스트레이션에 신규 체크 연결

**Files:**
- Modify: `backend/app/core/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 3의 `check_sensitive_words`, Task 5의 `run_panel(kr_audio_path=...)`,
  Task 6의 `check_accent_conformance`
- Produces: `QCPipeline.run()` 동작 확장 (시그니처 변경 없음)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_pipeline.py` 파일 끝에 추가 (기존 `job_files` fixture 재사용):

```python
async def test_pipeline_includes_sensitive_word_findings(job_files, monkeypatch, tmp_path):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    en, kr, stem = job_files
    # 사전에 확실히 걸리는 단어를 영어 SRT에 심는다
    sensitive_srt = tmp_path / "en_sensitive.srt"
    sensitive_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nthis line has PLACEHOLDER-SLUR-1 in it\n",
        encoding="utf-8",
    )
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=str(sensitive_srt), kr_srt_path=kr,
        stem_audio_path=stem,
    ))
    sensitive_findings = [f for f in result.findings if f.finding_type == "sensitive"]
    assert len(sensitive_findings) >= 1


async def test_pipeline_passes_kr_audio_path_to_panel_for_director(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    en, kr, stem = job_files
    pipeline = QCPipeline(provider=get_provider())
    # kr_audio_path 없이도(한국어 SRT만 제공) 예외 없이 완료되어야 한다 —
    # 원본 오디오가 없으면 그냥 클립 없이 진행(우아한 저하)
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=en, kr_srt_path=kr, stem_audio_path=stem,
    ))
    assert result.verdict.status in ("pass", "conditional", "fail")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — 민감어 finding이 아직 파이프라인에 연결되지 않아
`test_pipeline_includes_sensitive_word_findings`가 실패

- [ ] **Step 3: pipeline.py 수정**

`backend/app/core/pipeline.py`의 import 블록에서:
```python
from app.core.rule_checks import run_text_checks, check_audio_quality, check_srt_audio_match
```
을 다음으로 교체:
```python
from app.core.rule_checks import (
    run_text_checks, check_audio_quality, check_srt_audio_match, check_sensitive_words,
)
from app.core.accent import check_accent_conformance
```

`run` 메서드 안의 다음 블록:
```python
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
```
을 다음으로 교체:
```python
        # ③ 결정론적 룰 체크
        notify("rules", 0, 1)
        findings = run_text_checks(pairs) + check_sensitive_words(pairs)
        if job.stem_audio_path:
            findings += check_audio_quality(job.stem_audio_path, pairs)
            findings += await check_srt_audio_match(pairs, job.stem_audio_path, provider)
            findings += check_accent_conformance(pairs, job.stem_audio_path)
        notify("rules", 1, 1)

        # ④ 페르소나 패널 (연출가에게 원본 오디오도 함께 전달)
        scenes = group_by_scene(pairs)
        panel_findings = await run_panel(
            scenes, load_knowledge(), provider,
            stem_wav_path=job.stem_audio_path,
            kr_audio_path=job.kr_audio_path,
            on_progress=lambda d, t: notify("panel", d, t),
        )
        findings += panel_findings
```

- [ ] **Step 4: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: 파이프라인에 민감어 체크·억양 적합성 체크·연출가 원본 오디오 연결"
```

---

### Task 8: 프론트엔드 — 6축 표시 + 콘텐츠 플래그 섹션

**Files:**
- Modify: `frontend/src/views/ReportView.jsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: Task 1의 `QCFinding.finding_type` (JSON에서 `finding.finding_type`으로 노출됨)
- Produces: `ReportView`가 `finding_type === "sensitive"`인 finding을 별도
  섹션으로 렌더링하고, 수정 지시서 표에서는 제외

**참고:** 자동화 프론트엔드 테스트가 없는 프로젝트이므로, 검증은 `npm run build`
성공과 아래 수동 확인으로 한다.

- [ ] **Step 1: ReportView.jsx 수정**

`frontend/src/views/ReportView.jsx`에서 다음 줄:
```javascript
  // 검수자가 반려(오탐)한 finding 제외 = 확정 지시서
  const confirmed = findings.filter((f) => reviewed[f.id]?.action !== "rejected");
```
을 다음으로 교체:
```javascript
  // 검수자가 반려(오탐)한 finding 제외 = 확정 지시서 (민감어는 별도 섹션이라 제외)
  const confirmed = findings.filter(
    (f) => reviewed[f.id]?.action !== "rejected" && f.finding_type !== "sensitive"
  );
  const contentFlags = findings.filter(
    (f) => f.finding_type === "sensitive" && reviewed[f.id]?.action !== "rejected"
  );
```

다음 줄:
```javascript
      <h3>5축 MOS 스코어카드</h3>
```
을 다음으로 교체:
```javascript
      <h3>축별 MOS 스코어카드</h3>
```

`</table>` 바로 다음 줄(수정 지시서 테이블 뒤, `{reverdictError &&` 앞)에 삽입:
```javascript
      {contentFlags.length > 0 && (
        <div className="content-flags-section">
          <h3>⚠ 콘텐츠 플래그 ({contentFlags.length}건)</h3>
          <p className="content-flags-note">
            민감어·욕설 등 심의/등급에 영향을 줄 수 있는 표현입니다. MOS 점수에는
            반영되지 않으며, 판정 상태와 무관하게 검수자 확인이 필요합니다.
          </p>
          <table className="report-table content-flags-table">
            <thead>
              <tr><th>타임코드</th><th>유형</th><th>더빙 대사</th><th>지적 사유</th></tr>
            </thead>
            <tbody>
              {contentFlags.map((f) => (
                <tr key={f.id}>
                  <td>{f.start_time.toFixed(1)}s</td>
                  <td>{f.issue_type}</td>
                  <td>{f.current_translation}</td>
                  <td>{f.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
```

- [ ] **Step 2: App.css에 스타일 추가**

`frontend/src/App.css` 파일 끝에 추가:
```css
.content-flags-section { margin: 24px 0; padding: 16px; border-radius: 10px;
  background: #2b1414; border: 1px solid #5a2a2a; }
.content-flags-note { font-size: 13px; color: #d0a0a0; margin-bottom: 12px; }
.content-flags-table th, .content-flags-table td { border-bottom-color: #4a2424; }
```

- [ ] **Step 3: 빌드 검증**

Run: `cd frontend && npm run build`
Expected: 빌드 성공 (0 에러)

- [ ] **Step 4: 백엔드 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS (프론트엔드만 변경했지만 전체 스택 무결성 재확인)

- [ ] **Step 5: 커밋**

```bash
git add frontend/src/views/ReportView.jsx frontend/src/App.css
git commit -m "feat: 리포트 뷰에 콘텐츠 플래그 섹션 추가, MOS 축 개수 표기 일반화"
```

---

## 스펙 커버리지 노트 (자체 검토 결과)

- §5.1 언어 적합성, §5.4 자연스러움: v1 그대로, 변경 태스크 없음(의도됨) ✓
- §5.2 감정 표현(원본 오디오 대조): Task 5 ✓
- §5.3 싱크 정확도: **이미 v1에 구현되어 있어 태스크 불필요** — `check_sync_overflow`가
  타임코드 격차 기반으로 이미 동작 중임을 확인 (자체 검토 중 발견, 설계 스펙이
  "변경" 태스크로 잘못 표시했던 부분을 계획에서 바로잡음)
- §5.5 음질: v1 그대로, 변경 없음 (의도됨) ✓
- §5.6 억양 적합성: Task 6 ✓
- §5.7 민감어·욕설: Task 3(사전) + Task 4(원어민 통합) ✓
- §6.1 씬 배치 크기 상한: Task 2 ✓
- §7 컴포넌트 표의 `finding_type` 라우팅: Task 1(스키마/검증) + Task 8(프론트 표시) ✓
- **스펙 문서의 "MOS 5축→7축" 표현은 부정확했음** — 실제로는 MOS 6축(억양 적합성
  추가) + 별도 콘텐츠 플래그 1종(민감어). Global Constraints에 명시하고 Task 1에서
  정확히 6개로 구현.
