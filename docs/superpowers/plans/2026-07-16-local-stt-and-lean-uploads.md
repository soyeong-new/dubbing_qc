# 로컬 STT 도입 + 업로드 경량화 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 한국어 원본 오디오 STT를 Gemini(클라우드, 응답 길이 제한으로 장편 영화에서
잘림)에서 로컬 Whisper 파인튜닝 모델로 전환하고, STT/LLM 없이 순수 신호처리로 원본·더빙
오디오의 발화 타이밍 동기화를 확인하는 체크로 `check_srt_audio_match`를 재설계하며,
검수 탭의 중복 업로드를 제거한다.

**Architecture:** 새 `core/local_stt.py` 모듈이 `transformers` 파이프라인으로 로컬
Whisper 모델을 지연 로드해 한국어 STT를 전담한다. `ingest.py`는 이 모듈을 호출하도록
바뀌고 더 이상 `provider`가 필요 없다. `rule_checks.py`의 `check_srt_audio_match`(STT
기반 내용 비교)는 `check_dialogue_timing_sync`(RMS 기반 발화 시작점 비교)로 대체된다.
두 변경이 끝나면 `ModelProvider.transcribe()`를 호출하는 곳이 코드베이스에 하나도
남지 않아 인터페이스에서 완전히 제거한다.

**Tech Stack:** `transformers`(신규, 이미 설치된 torch 재사용), 기존 FastAPI/pytest 백엔드,
React 프론트엔드(자동화 테스트 없음, `npm run build`로 검증).

## Global Constraints

- 로컬 STT 모델: `batiai/batisay-ko-turbo` (Whisper Large-v3 한국어 파인튜닝, SafeTensors,
  `transformers` 라이브러리로 로드)
- 디바이스 선택: `torch.backends.mps.is_available()`가 참이면 `"mps"`, 아니면 `"cpu"`
- 긴 오디오 청킹은 `transformers` 파이프라인의 `chunk_length_s=30`, `stride_length_s=5`,
  `return_timestamps=True` 내장 기능을 그대로 사용 — 직접 파일을 자르는 로직을 만들지 않는다
- 실제 모델은 함수 내부에서 지연 로드하고, 자동화 테스트는 절대 실제 모델을 로드하지
  않는다 — `transcribe_fn`/`extract_clip_fn` 같은 주입 가능한 함수 파라미터로 대체
- 화자 라벨은 다루지 않는다 — `SegmentText.speaker`는 기본값 `"?"` 그대로 둔다
- `QCFinding.description`은 반드시 한국어, `recommendation`은 반드시 영어 (이 규칙은
  이전 작업에서 두 번 위반되어 리뷰에서 잡힌 적이 있으니 새로 작성하는 모든 finding
  문자열에서 특히 주의)
- `check_dialogue_timing_sync`의 허용 오차는 0.5초 (`check_sync_overflow`와 동일)
- 블로킹 동기 호출(transformers 추론, ffmpeg subprocess)은 반드시 `asyncio.to_thread`로
  감싸 이벤트 루프를 막지 않는다 (이 코드베이스 전역 확립된 패턴)
- 단일 실패로 전체 QC 작업이 죽지 않도록, 오디오 기반 체크는 `try/except`로 감싸
  실패 시 로그만 남기고 계속 진행한다 (`check_accent_conformance` 통합 패턴과 동일)
- 이 저장소는 별도 브랜치/워크트리 없이 `main`에 직접 커밋한다 (기존 확립된 워크플로)

---

### Task 1: 로컬 한국어 STT 모듈 (`backend/app/core/local_stt.py`, 신규)

**Files:**
- Create: `backend/app/core/local_stt.py`
- Modify: `backend/requirements.txt`
- Test: `backend/tests/test_local_stt.py`

**Interfaces:**
- Consumes: 없음 (기반 태스크)
- Produces: `transcribe_korean(audio_path: str, transcribe_fn: Optional[Callable[[str], list]] = None) -> List[SegmentText]`
  — Task 2가 이 함수를 가져다 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_local_stt.py` (신규 파일):

```python
from app.core.local_stt import transcribe_korean


def test_transcribe_korean_uses_injected_fn_and_parses_chunks():
    def fake_transcribe_fn(audio_path):
        assert audio_path == "/tmp/original.wav"
        return [
            {"text": " 형, 밥 먹었어?", "timestamp": (1.0, 3.0)},
            {"text": " 어이가 없네.", "timestamp": (5.2, 7.8)},
        ]

    segments = transcribe_korean("/tmp/original.wav", transcribe_fn=fake_transcribe_fn)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 3.0
    assert segments[0].text == "형, 밥 먹었어?"
    assert segments[1].text == "어이가 없네."


def test_transcribe_korean_skips_empty_chunks():
    def fake_transcribe_fn(audio_path):
        return [
            {"text": "   ", "timestamp": (0.0, 1.0)},
            {"text": "실제 대사", "timestamp": (1.0, 2.0)},
        ]

    segments = transcribe_korean("/tmp/x.wav", transcribe_fn=fake_transcribe_fn)
    assert len(segments) == 1
    assert segments[0].text == "실제 대사"


def test_transcribe_korean_handles_open_ended_last_chunk():
    # Whisper의 마지막 청크는 종료 타임스탬프가 None일 수 있다
    def fake_transcribe_fn(audio_path):
        return [{"text": "마지막 대사", "timestamp": (10.0, None)}]

    segments = transcribe_korean("/tmp/x.wav", transcribe_fn=fake_transcribe_fn)
    assert len(segments) == 1
    assert segments[0].start == 10.0
    assert segments[0].end == 10.0  # end가 없으면 start로 대체


def test_transcribe_korean_does_not_import_transformers_at_module_load(monkeypatch):
    # transcribe_fn을 주입하면 실제 모델 로드 경로(_get_pipeline)가 전혀 호출되지 않아야 한다
    import app.core.local_stt as local_stt

    def boom():
        raise AssertionError("실제 모델을 로드하면 안 된다")

    monkeypatch.setattr(local_stt, "_get_pipeline", boom)
    segments = transcribe_korean(
        "/tmp/x.wav", transcribe_fn=lambda p: [{"text": "ok", "timestamp": (0.0, 1.0)}]
    )
    assert len(segments) == 1
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_local_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.local_stt'`

- [ ] **Step 3: `local_stt.py` 구현**

```python
from typing import Callable, List, Optional
from app.schemas import SegmentText

MODEL_ID = "batiai/batisay-ko-turbo"

_pipeline = None


def _select_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline
    _pipeline = pipeline(
        "automatic-speech-recognition", model=MODEL_ID,
        chunk_length_s=30, stride_length_s=5, device=_select_device(),
    )
    return _pipeline


def _run_pipeline(audio_path: str) -> list:
    """실제 로컬 Whisper 모델 호출부. transcribe_korean()의 기본 transcribe_fn이다.

    자동화 테스트는 이 함수를 절대 호출하지 않는다 — 항상 transcribe_fn을 주입해서
    실제 모델 로드를 피한다.
    """
    pipe = _get_pipeline()
    result = pipe(audio_path, return_timestamps=True)
    return result.get("chunks", [])


def transcribe_korean(
    audio_path: str,
    transcribe_fn: Optional[Callable[[str], list]] = None,
) -> List[SegmentText]:
    transcribe_fn = transcribe_fn or _run_pipeline
    chunks = transcribe_fn(audio_path)
    segments = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if not text:
            continue
        start, end = chunk["timestamp"]
        segments.append(SegmentText(
            start=float(start),
            end=float(end) if end is not None else float(start),
            text=text,
        ))
    return segments
```

- [ ] **Step 4: `requirements.txt`에 의존성 추가**

`backend/requirements.txt` 파일 끝에 추가:
```text
transformers>=4.40.0
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_local_stt.py -v`
Expected: 전체 PASS (4/4) — `transformers`를 실제로 import하지 않으므로 설치 여부와
무관하게 통과해야 한다. `_get_pipeline`은 지연 임포트라 모듈 로드 시점에는
`transformers`가 필요 없다.

- [ ] **Step 6: 전체 회귀 확인 및 커밋**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

```bash
git add backend/app/core/local_stt.py backend/requirements.txt backend/tests/test_local_stt.py
git commit -m "feat: 로컬 한국어 STT 모듈 추가 (batisay-ko-turbo, transformers)"
```

---

### Task 2: `ingest.py`를 로컬 STT로 연동

**Files:**
- Modify: `backend/app/core/ingest.py`
- Test: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: Task 1의 `local_stt.transcribe_korean(audio_path, transcribe_fn=None) -> List[SegmentText]`
- Produces: `load_text_source(lang: str, srt_path: Optional[str], audio_path: Optional[str]) -> List[SegmentText]`
  — `provider` 파라미터가 완전히 제거된 새 시그니처. Task 4가 이 새 시그니처로 호출부를 갱신한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_ingest.py` 전체를 다음으로 교체:

```python
import pytest
from app.core.ingest import parse_srt, load_text_source
from app.schemas import SegmentText

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


async def test_load_text_source_prefers_srt(tmp_path):
    srt = tmp_path / "en.srt"
    srt.write_text(SAMPLE_SRT, encoding="utf-8")
    segments = await load_text_source("en", str(srt), "/tmp/audio.wav")
    assert segments[0].text == "Hey man, did you eat rice?"  # 로컬 STT가 아닌 SRT 결과


async def test_load_text_source_falls_back_to_local_stt_for_korean(monkeypatch):
    def fake_transcribe_korean(audio_path):
        return [SegmentText(start=0.0, end=1.0, text="눈치 좀 봐라")]

    monkeypatch.setattr("app.core.local_stt.transcribe_korean", fake_transcribe_korean)
    segments = await load_text_source("ko", None, "/tmp/audio.wav")
    assert "눈치" in segments[0].text


async def test_load_text_source_requires_some_input():
    with pytest.raises(ValueError):
        await load_text_source("ko", None, None)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: FAIL — `load_text_source()`가 여전히 `provider` 인자를 요구해 `TypeError`

- [ ] **Step 3: `ingest.py` 수정**

`backend/app/core/ingest.py` 전체를 다음으로 교체:

```python
import asyncio
import re
from typing import List, Optional
from app.schemas import SegmentText

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
                           audio_path: Optional[str]) -> List[SegmentText]:
    if srt_path:
        with open(srt_path, encoding="utf-8-sig") as f:
            return parse_srt(f.read())
    if audio_path and lang == "ko":
        from app.core.local_stt import transcribe_korean
        # transcribe_korean은 transformers 파이프라인을 동기로 호출한다 — asyncio
        # 이벤트 루프를 막지 않도록 스레드로 넘긴다.
        return await asyncio.to_thread(transcribe_korean, audio_path)
    raise ValueError(f"{lang}: SRT 또는 지원되는 오디오 STT 경로가 필요합니다.")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: 전체 PASS (5/5)

- [ ] **Step 5: 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 여기서는 FAIL이 예상됨 — `pipeline.py`가 아직 옛 시그니처(`provider` 인자
포함)로 `load_text_source`를 호출하고 있다. 이 실패는 Task 4에서 `pipeline.py`를
갱신하면 해결된다. 지금은 `test_ingest.py`와 `test_local_stt.py`만 통과하면 된다:

Run: `cd backend && venv/bin/python -m pytest tests/test_ingest.py tests/test_local_stt.py -v`
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add backend/app/core/ingest.py backend/tests/test_ingest.py
git commit -m "feat: ingest.py 한국어 STT 폴백을 로컬 STT로 전환, provider 파라미터 제거"
```

(참고: 이 커밋 시점에는 `pipeline.py`가 아직 옛 시그니처를 호출하므로 전체 스위트가
깨진 상태다. Task 4가 끝나야 다시 초록이 된다 — 이는 계획상 의도된 순서다.)

---

### Task 3: `check_dialogue_timing_sync` 추가 (기존 `check_srt_audio_match`는 유지)

**Files:**
- Modify: `backend/app/core/rule_checks.py`
- Test: `backend/tests/test_audio_checks.py`

**Interfaces:**
- Consumes: 없음 (기존 `read_wav_mono`, `_rms`, `_finding`, `extract_clip`를 같은 파일
  내에서 재사용)
- Produces: `check_dialogue_timing_sync(pairs, kr_audio_path, stem_audio_path, extract_clip_fn=None, tolerance=0.5, padding=0.5) -> List[QCFinding]`
  — Task 4가 이 함수로 `check_srt_audio_match`를 대체한다. `extract_clip_fn` 기본값은
  `None`이며 함수 본문에서 `extract_clip_fn or extract_clip`로 해석한다 — 이렇게 해야
  `monkeypatch.setattr("app.core.rule_checks.extract_clip", ...)`가 실제로 적용된다
  (기본 인자값으로 `extract_clip`을 직접 바인딩하면 정의 시점에 고정돼 나중에
  monkeypatch해도 반영되지 않는다 — `accent.py`의 `check_accent_conformance`가 이미
  쓰는 패턴과 동일).

**중요:** 이 태스크는 새 함수를 **추가만** 한다. 기존 `check_srt_audio_match`와 그 재시도
로직은 아직 삭제하지 않는다 — `pipeline.py`가 여전히 그걸 호출하고 있어서, 지금 지우면
Task 4 전까지 전체 스위트가 깨진다. 삭제는 Task 4에서 호출부를 새 함수로 옮긴 직후에
같이 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_audio_checks.py` 파일 끝에 추가:

```python
def test_check_dialogue_timing_sync_flags_offset_onset(tmp_path):
    from app.core.rule_checks import check_dialogue_timing_sync

    # 원본: 0.5초 무음 후 발화 시작
    kr_clip = tmp_path / "kr_clip.wav"
    write_wav(kr_clip, [0] * 8000 + sine(1.0))
    # 더빙: 1.3초 무음 후 발화 시작 (원본 대비 0.8초 밀림)
    en_clip = tmp_path / "en_clip.wav"
    write_wav(en_clip, [0] * 20800 + sine(1.0))

    def fake_extract(src, start, end):
        return str(kr_clip) if src == "kr_original.wav" else str(en_clip)

    pairs = [pair_at(1.0, 3.0, pid="p1")]
    findings = check_dialogue_timing_sync(
        pairs, "kr_original.wav", "en_stem.wav",
        extract_clip_fn=fake_extract, tolerance=0.5,
    )
    assert len(findings) == 1
    assert findings[0].issue_type == "발화 타이밍 불일치"
    assert findings[0].axis == "싱크 정확도"
    assert findings[0].recommendation.isascii()  # recommendation은 반드시 영어


def test_check_dialogue_timing_sync_passes_when_aligned(tmp_path):
    from app.core.rule_checks import check_dialogue_timing_sync

    # 원본/더빙 모두 0.5초 무음 후 발화 시작 (차이 없음)
    clip = tmp_path / "aligned_clip.wav"
    write_wav(clip, [0] * 8000 + sine(1.0))

    findings = check_dialogue_timing_sync(
        pairs=[pair_at(1.0, 3.0, pid="p1")],
        kr_audio_path="kr.wav", stem_audio_path="en.wav",
        extract_clip_fn=lambda src, s, e: str(clip), tolerance=0.5,
    )
    assert findings == []


def test_check_dialogue_timing_sync_skips_when_either_side_missing():
    from app.core.rule_checks import check_dialogue_timing_sync
    from app.schemas import AlignedPair, SegmentText

    pairs = [AlignedPair(
        id="p1", korean=None,
        dubbed=SegmentText(start=0, end=1, speaker="A", text="hi"),
    )]
    findings = check_dialogue_timing_sync(
        pairs, "kr.wav", "en.wav", extract_clip_fn=lambda s, a, b: s,
    )
    assert findings == []


def test_check_dialogue_timing_sync_skips_when_no_speech_detected(tmp_path):
    from app.core.rule_checks import check_dialogue_timing_sync

    silent_clip = tmp_path / "silent.wav"
    write_wav(silent_clip, [0] * 16000)  # 완전 무음

    findings = check_dialogue_timing_sync(
        pairs=[pair_at(0.0, 1.0, pid="p1")],
        kr_audio_path="kr.wav", stem_audio_path="en.wav",
        extract_clip_fn=lambda src, s, e: str(silent_clip),
    )
    assert findings == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_audio_checks.py -v -k timing_sync`
Expected: FAIL — `ImportError: cannot import name 'check_dialogue_timing_sync'`

- [ ] **Step 3: `rule_checks.py`에 함수 추가**

`backend/app/core/rule_checks.py`의 `check_srt_audio_match` 함수 바로 다음 줄에
(즉 `_DEFAULT_SENSITIVE_WORDS = ...` 줄 앞에) 추가:

```python
def _find_speech_onset(samples, rate, threshold: float = 100, frame_ms: int = 100):
    """구간 내에서 무음이 아닌(발화가 시작되는) 첫 프레임의 시각(초)을 반환한다.
    발화가 감지되지 않으면 None을 반환한다."""
    frame = max(1, int(rate * frame_ms / 1000))
    for i in range(0, len(samples), frame):
        if _rms(samples[i:i + frame]) >= threshold:
            return i / rate
    return None


def check_dialogue_timing_sync(
    pairs: List[AlignedPair], kr_audio_path: str, stem_audio_path: str,
    extract_clip_fn=None, tolerance: float = 0.5, padding: float = 0.5,
) -> List[QCFinding]:
    """원본과 더빙, 두 오디오 트랙에서 실제 발화가 같은 시간대에 시작되는지 확인한다.

    자막 내용의 의미가 같은지는 3-페르소나 패널이 오디오를 직접 들으며 이미 판단하므로
    여기서는 다루지 않는다 — 순수하게 발화 타이밍만 신호처리로 비교한다.
    """
    # 기본값을 extract_clip_fn=extract_clip처럼 직접 바인딩하면 정의 시점에 고정돼
    # 나중에 monkeypatch.setattr("app.core.rule_checks.extract_clip", ...)로 테스트에서
    # 갈아끼워도 반영되지 않는다 — 반드시 호출 시점에 지연 평가해야 한다.
    extract_clip_fn = extract_clip_fn or extract_clip
    findings = []
    for p in pairs:
        if not p.korean or not p.dubbed:
            continue
        kr_window_start = max(0.0, p.korean.start - padding)
        en_window_start = max(0.0, p.dubbed.start - padding)
        kr_clip = extract_clip_fn(kr_audio_path, kr_window_start, p.korean.end + padding)
        en_clip = extract_clip_fn(stem_audio_path, en_window_start, p.dubbed.end + padding)
        kr_samples, kr_rate = read_wav_mono(kr_clip)
        en_samples, en_rate = read_wav_mono(en_clip)
        kr_onset = _find_speech_onset(kr_samples, kr_rate)
        en_onset = _find_speech_onset(en_samples, en_rate)
        if kr_onset is None or en_onset is None:
            continue
        kr_global = kr_window_start + kr_onset
        en_global = en_window_start + en_onset
        diff = abs(kr_global - en_global)
        if diff > tolerance:
            findings.append(_finding(
                "timingsync", p, "medium", "발화 타이밍 불일치", "싱크 정확도",
                f"원본과 더빙 오디오의 실제 발화 시작 시점이 {diff:.2f}초 차이납니다.",
                "Re-check the dubbed audio timing against the original track.",
                category="voice",
            ))
    return findings
```

- [ ] **Step 4: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_audio_checks.py -v`
Expected: 전체 PASS (기존 `check_srt_audio_match` 테스트들도 여전히 통과 — 아직
삭제하지 않았으므로)

Run: `cd backend && venv/bin/python -m pytest tests/test_ingest.py tests/test_local_stt.py tests/test_audio_checks.py -v`
Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add backend/app/core/rule_checks.py backend/tests/test_audio_checks.py
git commit -m "feat: check_dialogue_timing_sync 추가 (RMS 기반 발화 타이밍 비교, STT 불필요)"
```

---

### Task 4: 파이프라인 재연결 + 옛 STT 기반 체크 삭제

**Files:**
- Modify: `backend/app/core/pipeline.py`
- Modify: `backend/app/core/rule_checks.py`
- Modify: `backend/tests/test_audio_checks.py`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 2의 `load_text_source(lang, srt_path, audio_path)` (provider 없는 새 시그니처),
  Task 3의 `check_dialogue_timing_sync(pairs, kr_audio_path, stem_audio_path, ...)`
- Produces: 없음 (오케스트레이션 배선의 끝단)

- [ ] **Step 1: 실패하는 테스트 작성**

`backend/tests/test_pipeline.py` 파일 끝에 추가:

```python
async def test_pipeline_runs_dialogue_timing_sync_when_both_audio_present(job_files, monkeypatch, tmp_path):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setattr("app.core.accent.classify_accent", _fake_classify_accent)
    en, kr, stem = job_files
    # 원본 오디오도 있어야 발화 타이밍 체크가 돈다 (job_files 픽스처는 stem만 제공)
    kr_audio = tmp_path / "kr_audio.wav"
    rate, samples = 16000, []
    for i in range(rate * 7):
        samples.append(int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
    with wave.open(str(kr_audio), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack(f"{len(samples)}h", *samples))

    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=en, kr_srt_path=kr,
        kr_audio_path=str(kr_audio), stem_audio_path=stem,
    ))
    # 예외 없이 완료되면 충분하다 — 두 트랙 모두 같은 사인파라 실제로 타이밍이
    # 어긋날 이유가 없어 finding이 없어도 정상이다. 여기서는 "두 오디오가 모두 있을 때
    # 크래시 없이 파이프라인이 이 체크를 실행한다"는 배선 자체를 검증한다.
    assert result.verdict.status in ("pass", "conditional", "fail")


async def test_pipeline_survives_dialogue_timing_sync_failure(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setattr("app.core.accent.classify_accent", _fake_classify_accent)
    en, kr, stem = job_files

    def raise_extract(src, start, end):
        raise RuntimeError("ffmpeg 실패")

    monkeypatch.setattr("app.core.rule_checks.extract_clip", raise_extract)
    pipeline = QCPipeline(provider=get_provider())
    # kr_audio_path를 stem과 같은 파일로 재사용해 체크가 시도되게 하되, extract_clip이
    # 실패하도록 몽키패치했으므로 우아하게 건너뛰어야 한다 (전체 파이프라인은 안 죽음).
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=en, kr_srt_path=kr,
        kr_audio_path=stem, stem_audio_path=stem,
    ))
    assert all(f.issue_type != "발화 타이밍 불일치" for f in result.findings)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_pipeline.py -v -k timing_sync`
Expected: FAIL — `pipeline.py`가 아직 `check_dialogue_timing_sync`를 호출하지 않음
(또한 이 시점에는 `load_text_source` 시그니처 불일치로 전체 스위트가 이미 깨져 있음 —
Task 2의 Step 5 참고)

- [ ] **Step 3: `pipeline.py` 수정**

`backend/app/core/pipeline.py`에서:
```python
from app.core.rule_checks import (
    run_text_checks, check_audio_quality, check_srt_audio_match, check_sensitive_words,
)
```
를 다음으로 교체:
```python
from app.core.rule_checks import (
    run_text_checks, check_audio_quality, check_dialogue_timing_sync, check_sensitive_words,
)
```

다음 블록:
```python
        # ① 텍스트 수집 (SRT 우선, STT 폴백)
        notify("ingest", 0, 2)
        korean = await load_text_source("ko", job.kr_srt_path, job.kr_audio_path, provider)
        notify("ingest", 1, 2)
        dubbed = await load_text_source("en", job.en_srt_path, None, provider)
        notify("ingest", 2, 2)
```
를 다음으로 교체:
```python
        # ① 텍스트 수집 (SRT 우선, STT 폴백)
        notify("ingest", 0, 2)
        korean = await load_text_source("ko", job.kr_srt_path, job.kr_audio_path)
        notify("ingest", 1, 2)
        dubbed = await load_text_source("en", job.en_srt_path, None)
        notify("ingest", 2, 2)
```

다음 블록:
```python
        # ③ 결정론적 룰 체크
        notify("rules", 0, 1)
        findings = run_text_checks(pairs) + check_sensitive_words(pairs)
        if job.stem_audio_path:
            findings += check_audio_quality(job.stem_audio_path, pairs)
            findings += await check_srt_audio_match(pairs, job.stem_audio_path, provider)
            try:
                # check_accent_conformance는 세그먼트마다 ffmpeg 클립 추출(동기 subprocess)과
                # SpeechBrain 추론을 동기로 수행한다 — asyncio 이벤트 루프를 막지 않도록
                # 스레드로 넘긴다 (그렇지 않으면 진행률 폴링 등 다른 요청이 이 억양 분류가
                # 끝날 때까지 전부 멈춘다).
                findings += await asyncio.to_thread(
                    check_accent_conformance, pairs, job.stem_audio_path
                )
            except Exception as e:
                # 클립 추출 실패, 모델 로드 실패 등 단일 원인으로 전체 QC 작업이
                # 죽지 않도록 억양 체크만 건너뛰고 계속 진행한다 (우아한 저하) —
                # 이미 계산된 텍스트/오디오 findings를 버리는 것보다 낫다.
                print(f"[파이프라인] 억양 분류 실패, 해당 체크 없이 진행: {e}")
        notify("rules", 1, 1)
```
를 다음으로 교체:
```python
        # ③ 결정론적 룰 체크
        notify("rules", 0, 1)
        findings = run_text_checks(pairs) + check_sensitive_words(pairs)
        if job.stem_audio_path:
            findings += check_audio_quality(job.stem_audio_path, pairs)
            if job.kr_audio_path:
                try:
                    # check_dialogue_timing_sync는 세그먼트마다 ffmpeg 클립 추출(동기
                    # subprocess)과 RMS 신호 분석을 동기로 수행한다 — asyncio 이벤트
                    # 루프를 막지 않도록 스레드로 넘긴다.
                    findings += await asyncio.to_thread(
                        check_dialogue_timing_sync, pairs, job.kr_audio_path, job.stem_audio_path
                    )
                except Exception as e:
                    print(f"[파이프라인] 발화 타이밍 동기화 체크 실패, 해당 체크 없이 진행: {e}")
            try:
                # check_accent_conformance는 세그먼트마다 ffmpeg 클립 추출(동기 subprocess)과
                # SpeechBrain 추론을 동기로 수행한다 — asyncio 이벤트 루프를 막지 않도록
                # 스레드로 넘긴다 (그렇지 않으면 진행률 폴링 등 다른 요청이 이 억양 분류가
                # 끝날 때까지 전부 멈춘다).
                findings += await asyncio.to_thread(
                    check_accent_conformance, pairs, job.stem_audio_path
                )
            except Exception as e:
                # 클립 추출 실패, 모델 로드 실패 등 단일 원인으로 전체 QC 작업이
                # 죽지 않도록 억양 체크만 건너뛰고 계속 진행한다 (우아한 저하) —
                # 이미 계산된 텍스트/오디오 findings를 버리는 것보다 낫다.
                print(f"[파이프라인] 억양 분류 실패, 해당 체크 없이 진행: {e}")
        notify("rules", 1, 1)
```

- [ ] **Step 4: 옛 `check_srt_audio_match`와 그 테스트 삭제**

`backend/app/core/rule_checks.py`에서 다음 함수 전체를 삭제한다 (`extract_clip` 함수
바로 다음, `_DEFAULT_SENSITIVE_WORDS` 줄 바로 앞에 있음):

```python
async def check_srt_audio_match(pairs: List[AlignedPair], stem_wav_path: str,
                                provider: ModelProvider, extract_clip_fn=extract_clip,
                                sample_every: int = 10) -> List[QCFinding]:
    """Check if dubbed audio matches the SRT text via transcription."""
    findings = []
    targets = [p for p in pairs if p.dubbed and p.dubbed.text.strip()][::sample_every]
    for p in targets:
        # extract_clip_fn은 ffmpeg를 동기 호출한다 — asyncio 이벤트 루프를
        # 막지 않도록 스레드로 넘긴다.
        heard = None
        for attempt in (1, 2):
            try:
                clip = await asyncio.to_thread(
                    extract_clip_fn, stem_wav_path, p.dubbed.start, p.dubbed.end)
                heard = await provider.transcribe(clip, lang="en")
                break
            except Exception as e:
                # STT 응답이 깨진 JSON이거나 클립 추출이 실패해도 한 세그먼트
                # 때문에 전체 QC가 죽으면 안 된다 — 1회 재시도 후 건너뛴다.
                if attempt == 2:
                    print(f"[룰체크] {p.id} 자막-음성 대조 실패 (2회 시도), 건너뜀: {e}")
        if heard is None:
            continue
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

같은 파일 상단의 import 블록에서:
```python
import asyncio
import math
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List
import yaml
from app.schemas import AlignedPair, QCFinding
from app.providers.base import ModelProvider
```
를 다음으로 교체 (`asyncio` import와 `ModelProvider` import가 이제 이 파일에서 쓰이지
않으므로 함께 제거 — `asyncio`는 삭제된 `check_srt_audio_match`에서만 쓰였다):
```python
import math
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List
import yaml
from app.schemas import AlignedPair, QCFinding
```

`backend/tests/test_audio_checks.py`에서 다음 세 테스트 함수 전체를 삭제한다:
- `test_srt_audio_match_flags_mismatch`
- `test_srt_audio_match_retries_once_on_transcribe_failure`
- `test_srt_audio_match_skips_segment_after_persistent_failure`

같은 파일 상단의:
```python
from app.core.rule_checks import (
    read_wav_mono, check_audio_quality, check_srt_audio_match, _token_similarity,
)
from app.providers.base import get_provider
```
를 다음으로 교체 (`check_srt_audio_match`가 삭제됐고, 삭제된 테스트들만 쓰던
`get_provider` import도 더 이상 필요 없다 — `_token_similarity`는 여전히 독립적으로
테스트되므로 유지):
```python
from app.core.rule_checks import read_wav_mono, check_audio_quality, _token_similarity
```

- [ ] **Step 5: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_pipeline.py tests/test_audio_checks.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS — Task 2에서 남겨뒀던 시그니처 불일치가 여기서 해소된다.

- [ ] **Step 6: 커밋**

```bash
git add backend/app/core/pipeline.py backend/app/core/rule_checks.py backend/tests/test_pipeline.py backend/tests/test_audio_checks.py
git commit -m "feat: 파이프라인을 로컬 STT/발화 타이밍 체크로 재배선, 옛 STT 기반 자막-음성 대조 삭제"
```

---

### Task 5: `ModelProvider`에서 `transcribe()` 완전 제거

**Files:**
- Modify: `backend/app/providers/base.py`
- Modify: `backend/app/providers/gemini.py`
- Modify: `backend/app/providers/mock.py`
- Modify: `backend/tests/test_providers.py`
- Modify: `backend/tests/test_judge_panel.py`

**Interfaces:**
- Consumes: 없음 (Task 4가 끝난 시점에 `transcribe()`를 호출하는 곳이 코드베이스에
  하나도 없음을 전제)
- Produces: 없음 (인터페이스 축소)

- [ ] **Step 1: 호출부가 정말 하나도 없는지 확인**

Run: `grep -rn "\.transcribe(" backend/app/ backend/tests/`
Expected: 결과가 나오지 않아야 한다 (Task 2, 4에서 이미 다 제거됨). 만약 결과가
나온다면 이 태스크를 진행하지 말고 컨트롤러에게 보고한다 — 이 단계는 반드시
사전 조건이 성립해야 안전하다.

- [ ] **Step 2: `base.py`에서 추상 메서드 제거**

`backend/app/providers/base.py`에서:
```python
class ModelProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        ...

    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        ...
```
를 다음으로 교체:
```python
class ModelProvider(ABC):
    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        ...
```

같은 파일에서 `SegmentText` import가 이제 쓰이지 않으면 제거한다 (파일 상단 import
줄을 확인해서 `SegmentText`가 다른 곳에 안 쓰이면 지운다):
```python
from app.schemas import SegmentText, AlignedPair, QCFinding
```
를 다음으로 교체:
```python
from app.schemas import AlignedPair, QCFinding
```

- [ ] **Step 3: `gemini.py`에서 구현 제거**

`backend/app/providers/gemini.py`에서 `GeminiProvider.transcribe` 메서드 전체를 삭제한다:
```python
    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        lang_name = "한국어" if lang == "ko" else "영어"
        audio_data = await asyncio.to_thread(_compress_to_mp3, audio_path)
        model = self._genai.GenerativeModel(MODEL_NAME)
        # google-generativeai의 generate_content는 동기(blocking) 호출이다.
        # 스레드로 넘기지 않으면 이 응답을 기다리는 동안 단일 asyncio 이벤트 루프
        # 전체가 멈춰, 같은 프로세스가 처리해야 할 진행률 폴링 요청까지 응답이 끊긴다.
        response = await asyncio.to_thread(
            model.generate_content,
            [{"mime_type": "audio/mp3", "data": audio_data},
             STT_PROMPT.format(lang_name=lang_name)],
            generation_config={"response_mime_type": "application/json"},
        )
        return parse_stt_response(response.text)
```

같은 파일에서 `STT_PROMPT` 상수와 `parse_stt_response` 함수도 이제 `transcribe()`에서만
쓰였으므로 함께 삭제한다:
```python
STT_PROMPT = """
제공된 {lang_name} 오디오 파일을 듣고, 화자별 대사와 시작/종료 시간(초)을 추출하십시오.
세그먼트는 발화 단위로 1~4초 내외로 분할하십시오.
반드시 아래 JSON 배열만 반환하십시오:
[{{"start": 1.2, "end": 4.5, "speaker": "인물 1", "text": "대사 내용"}}]
"""
```
전체 삭제.

```python
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
```
전체 삭제.

같은 파일 상단의:
```python
from app.schemas import SegmentText, AlignedPair, QCFinding, AXES
```
에서 `SegmentText`가 이제 이 파일 어디에도 쓰이지 않으면:
```python
from app.schemas import AlignedPair, QCFinding, AXES
```
로 교체한다.

- [ ] **Step 4: `mock.py`에서 구현 제거**

`backend/app/providers/mock.py`에서 `MockProvider.transcribe` 메서드 전체를 삭제한다:
```python
    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        return [
            SegmentText(start=1.0, end=4.5, speaker="화자1", text="임마, 너 어제 눈치 보며 기어 다녔다며?"),
            SegmentText(start=5.2, end=7.8, speaker="화자2", text="어이가 없네. 밥도 못 먹고 조사받고 있어요."),
        ]
```

같은 파일 상단의:
```python
from app.schemas import SegmentText, AlignedPair, QCFinding
```
에서 `SegmentText`가 이제 이 파일 어디에도 쓰이지 않으면:
```python
from app.schemas import AlignedPair, QCFinding
```
로 교체한다.

- [ ] **Step 5: 죽은 테스트 정리**

`backend/tests/test_providers.py`에서 `test_mock_transcribe_returns_segments` 함수
전체를 삭제한다:
```python
async def test_mock_transcribe_returns_segments(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    segments = await provider.transcribe("/tmp/nonexistent.wav", lang="ko")
    assert len(segments) >= 2
    assert segments[0].start < segments[1].start
```

`backend/tests/test_judge_panel.py`에서 `RecordingProvider` 클래스 안의 다음 스텁
메서드를 삭제한다 (더 이상 `ModelProvider` ABC가 요구하지 않으므로 불필요):
```python
        async def transcribe(self, audio_path, lang):
            return []

```
(바로 다음에 오는 `async def judge(...)` 메서드는 그대로 둔다.)

- [ ] **Step 6: 테스트 통과 및 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest tests/test_providers.py tests/test_judge_panel.py tests/test_gemini_provider.py -v`
Expected: 전체 PASS

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS

- [ ] **Step 7: 커밋**

```bash
git add backend/app/providers/base.py backend/app/providers/gemini.py backend/app/providers/mock.py backend/tests/test_providers.py backend/tests/test_judge_panel.py
git commit -m "refactor: ModelProvider에서 transcribe() 완전 제거 (더 이상 호출하는 곳 없음)"
```

---

### Task 6: 검수 탭 업로드 중복 제거

**Files:**
- Modify: `frontend/src/views/ProjectView.jsx`
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: 없음 (프론트엔드 전용, 백엔드 API 계약 변경 없음)
- Produces: 없음

**참고:** 이 프로젝트는 프론트엔드 자동화 테스트가 없다. 검증은 `npm run build` 성공과
아래 수동 확인으로 한다.

- [ ] **Step 1: `ProjectView.jsx`에서 업로드 시 File 객체도 저장**

`frontend/src/views/ProjectView.jsx`에서:
```javascript
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
```
를 다음으로 교체 (검수 탭이 서버에 재업로드하지 않고 로컬 재생용 blob URL을 만들 수
있도록 `file` 객체 자체를 상태에 함께 저장한다):
```javascript
  const handleFile = async (role, file) => {
    if (!file) return;
    setUploads((u) => ({ ...u, [role]: { name: file.name, uploading: true, file } }));
    const res = await uploadMedia(file, role);
    setUploads((u) => ({
      ...u,
      [role]: res.success
        ? { name: file.name, file, ...res, uploading: false }
        : { name: file.name, file, error: res.error, uploading: false },
    }));
  };
```

- [ ] **Step 2: `App.jsx`에서 중복 업로드 로직/상태 제거**

`frontend/src/App.jsx`에서 다음 상태 선언 블록:
```javascript
  // Video & audio states
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(10.0);
  const [originalVideoSrc, setOriginalVideoSrc] = useState(null);
  const [dubbedVideoSrc, setDubbedVideoSrc] = useState("https://assets.mixkit.co/videos/preview/mixkit-cyberpunk-city-street-at-night-40134-large.mp4");
  const [activeVideoRole, setActiveVideoRole] = useState("dubbed");

  // File names for display
  const [videoFileName, setVideoFileName] = useState("cyberpunk_city.mp4 (기본 샘플)");

  // Real waveform & backend audio path states
  const [waveformPeaks, setWaveformPeaks] = useState([]);
  const [backendAudioPath, setBackendAudioPath] = useState(null);
  const [uploadingVideo, setUploadingVideo] = useState(false);

  // Original media (Review 탭에서 원본 재생용 — QC 실행과는 무관)
  const [originalMediaName, setOriginalMediaName] = useState("선택되지 않음");
  const [originalAudioPath, setOriginalAudioPath] = useState(null);
  const [uploadingOriginal, setUploadingOriginal] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const videoInputRef = useRef(null);
  const originalInputRef = useRef(null);
```
를 다음으로 교체:
```javascript
  // Video & audio states
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(10.0);
  const DEFAULT_DUBBED_SAMPLE = "https://assets.mixkit.co/videos/preview/mixkit-cyberpunk-city-street-at-night-40134-large.mp4";
  const [originalVideoSrc, setOriginalVideoSrc] = useState(null);
  const [dubbedVideoSrc, setDubbedVideoSrc] = useState(DEFAULT_DUBBED_SAMPLE);
  const [activeVideoRole, setActiveVideoRole] = useState("dubbed");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
```

다음 함수 블록 전체(`handleOriginalUpload`, `handleVideoUpload`)를 삭제:
```javascript
  // 2. File Upload Handlers
  const handleOriginalUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setOriginalVideoSrc(url);
      setOriginalMediaName(file.name);
      setActiveVideoRole("original");
      setIsPlaying(false);
      if (videoRef.current) {
        videoRef.current.load();
      }

      setUploadingOriginal(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("http://localhost:8000/api/qc/upload-media?role=original", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.success) {
          setOriginalAudioPath(data.audio_path);
          console.log("Original media uploaded successfully:", data);
        } else {
          console.error("Original media upload failed:", data.error);
          setAnalysisError(`원본 미디어 업로드 실패: ${data.error || "알 수 없는 오류"}`);
        }
      } catch (err) {
        console.error("Error uploading original media:", err);
        setAnalysisError(`원본 미디어 업로드 실패: ${err.message}`);
      } finally {
        setUploadingOriginal(false);
      }
    }
  };

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setDubbedVideoSrc(url);
      setVideoFileName(file.name);
      setActiveVideoRole("dubbed");
      setIsPlaying(false);
      if (videoRef.current) {
        videoRef.current.load();
      }

      // Upload to backend to extract audio and compute waveform
      setUploadingVideo(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("http://localhost:8000/api/qc/upload-media?role=dubbed", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.success) {
          setWaveformPeaks(data.waveform);
          setBackendAudioPath(data.audio_path);
          console.log("Video uploaded and audio/waveform extracted successfully:", data);
        } else {
          console.error("Video upload failed:", data.error);
          setAnalysisError(`영상 업로드 실패: ${data.error || "알 수 없는 오류"}`);
        }
      } catch (err) {
        console.error("Error uploading video:", err);
        setAnalysisError(`영상 업로드 실패: ${err.message}`);
      } finally {
        setUploadingVideo(false);
      }
    }
  };

```
(이 블록 삭제 후, 바로 다음에 오는 `const updateStats = ...` 함수는 그대로 둔다.)

`updateStats` 함수 바로 앞, 삭제된 두 핸들러 자리에 다음 `useEffect` 두 개를 추가한다
(Project 탭에서 이미 받은 `uploads.original`/`uploads.dubbed`의 `file` 객체로부터
로컬 재생용 blob URL을 파생시킨다):
```javascript
  // 2. Video preview — Project 탭에서 이미 업로드된 File 객체로 로컬 blob URL만 생성한다.
  // 서버에 다시 업로드하지 않는다 (검수 탭 전용 중복 업로드였던 것을 제거).
  useEffect(() => {
    const file = uploads.original?.file;
    if (!file) return;
    const url = URL.createObjectURL(file);
    setOriginalVideoSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [uploads.original?.file]);

  useEffect(() => {
    const file = uploads.dubbed?.file;
    if (!file) return;
    const url = URL.createObjectURL(file);
    setDubbedVideoSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [uploads.dubbed?.file]);

```

- [ ] **Step 3: 헤더의 "미디어 등록" 버튼 UI 제거**

`frontend/src/App.jsx`에서 다음 블록:
```javascript
        {/* 재생용 미디어 등록 (Review 탭 전용 — QC 실행은 프로젝트 탭에서 이미 완료된 잡의 결과다) */}
        <div className="header-file-panel">
          {/* 1. Original KR Media */}
          <div className="file-uploader-box">
            <span className="file-label" title={originalMediaName}>🎙️ 원본 영상/음성 (KR): {originalMediaName}</span>
            <button className="btn-file-select" onClick={() => originalInputRef.current.click()} disabled={uploadingOriginal}>
              {uploadingOriginal ? "업로드 중..." : (originalAudioPath ? "등록 완료 ✓" : "미디어 등록")}
            </button>
            <input
              type="file"
              ref={originalInputRef}
              style={{ display: "none" }}
              accept="video/*,audio/*"
              onChange={handleOriginalUpload}
            />
          </div>

          {/* 2. Dubbed EN Media */}
          <div className="file-uploader-box">
            <span className="file-label" title={videoFileName}>🔊 영어 더빙 영상/음성: {videoFileName}</span>
            <button className="btn-file-select" onClick={() => videoInputRef.current.click()} disabled={uploadingVideo}>
              {uploadingVideo ? "분석 중..." : (backendAudioPath ? "등록 완료 ✓" : "미디어 등록")}
            </button>
            <input
              type="file"
              ref={videoInputRef}
              style={{ display: "none" }}
              accept="video/*,audio/*"
              onChange={handleVideoUpload}
            />
          </div>
        </div>
```
를 다음으로 교체 (버튼을 없애고, Project 탭에서 이미 등록된 파일명을 읽기 전용으로
보여준다):
```javascript
        {/* 재생 상태 표시 (Review 탭 전용 — 업로드는 프로젝트 탭에서 이미 완료됨) */}
        <div className="header-file-panel">
          <div className="file-uploader-box">
            <span className="file-label" title={uploads.original?.name}>
              🎙️ 원본 영상/음성 (KR): {uploads.original?.name || "선택되지 않음"}
            </span>
          </div>
          <div className="file-uploader-box">
            <span className="file-label" title={uploads.dubbed?.name}>
              🔊 영어 더빙 영상/음성: {uploads.dubbed?.name || "선택되지 않음"}
            </span>
          </div>
        </div>
```

- [ ] **Step 4: `waveformPeaks` 참조를 `uploads.dubbed?.waveform`으로 교체**

`frontend/src/App.jsx`에서 canvas 파형 렌더링 `useEffect` 안의:
```javascript
      if (waveformPeaks && waveformPeaks.length > 0) {
        // Draw real waveform
        const numPeaks = waveformPeaks.length;
        const barWidth = width / numPeaks;
        
        ctx.fillStyle = "rgba(168, 85, 247, 0.45)"; // Deep purple translucent
        for (let i = 0; i < numPeaks; i++) {
          const peak = waveformPeaks[i];
```
를 다음으로 교체:
```javascript
      const waveformPeaks = uploads.dubbed?.waveform;
      if (waveformPeaks && waveformPeaks.length > 0) {
        // Draw real waveform
        const numPeaks = waveformPeaks.length;
        const barWidth = width / numPeaks;
        
        ctx.fillStyle = "rgba(168, 85, 247, 0.45)"; // Deep purple translucent
        for (let i = 0; i < numPeaks; i++) {
          const peak = waveformPeaks[i];
```

같은 `useEffect`의 의존성 배열:
```javascript
  }, [isPlaying, currentTime, duration, waveformPeaks]);
```
를 다음으로 교체:
```javascript
  }, [isPlaying, currentTime, duration, uploads.dubbed?.waveform]);
```

- [ ] **Step 5: 빌드 검증**

Run: `cd frontend && npm run build`
Expected: 빌드 성공, 0 에러. `videoFileName`/`originalMediaName`/`waveformPeaks`(상태
변수)/`backendAudioPath`/`originalAudioPath`/`uploadingVideo`/`uploadingOriginal`/
`videoInputRef`/`originalInputRef`/`handleVideoUpload`/`handleOriginalUpload`에 대한
"정의되지 않음" 참조가 하나도 없어야 한다 — 빌드 로그를 눈으로 확인해서 미사용 변수
경고가 이전에 삭제한 것들을 가리키고 있지 않은지 확인.

- [ ] **Step 6: 백엔드 전체 회귀 확인**

Run: `cd backend && venv/bin/python -m pytest -q`
Expected: 전체 PASS (프론트엔드만 변경했지만 전체 스택 무결성 재확인)

- [ ] **Step 7: 커밋**

```bash
git add frontend/src/views/ProjectView.jsx frontend/src/App.jsx
git commit -m "refactor: 검수 탭의 중복 업로드 로직 제거, Project 탭 업로드 결과 재사용"
```

---

## 스펙 커버리지 노트 (자체 검토 결과)

- §3.1 로컬 한국어 STT: Task 1 ✓
- §3.2 `ingest.py` 연동: Task 2 ✓
- §3.3 `check_dialogue_timing_sync` 재설계: Task 3(추가) + Task 4(교체 및 옛 함수 삭제) ✓
  — 두 태스크로 나눈 이유는 "매 태스크마다 전체 스위트가 통과해야 한다"는 원칙과
  "한 태스크 안에서 옛 함수를 지우면서 동시에 호출부도 옮겨야 깨지지 않는다"는 제약을
  동시에 만족시키기 위함. Task 2에서도 같은 이유로 한 커밋 구간(Task 2 종료 시점)에서
  일시적으로 전체 스위트가 깨지는 것을 명시적으로 허용하고, Task 4에서 바로 잡는다.
- §3.4 `ModelProvider` 인터페이스 정리: Task 5 ✓ (사전조건을 Step 1에서 grep으로 재확인)
- §3.5 검수 탭 업로드 중복 제거: Task 6 ✓
- §4 테스트 전략: 모든 백엔드 태스크가 실제 모델/네트워크 없는 주입 가능한 함수 패턴을
  따름 (Task 1, 3). 프론트엔드는 스펙이 명시한 대로 `npm run build` + 수동 확인으로 검증.
- 스펙 §2에서 제외한 항목(영어 STT 폴백, 화자 분리, 배포 아키텍처, ffmpeg.wasm 클라이언트
  추출)은 이 계획의 어떤 태스크에도 포함되지 않았음 — 의도된 누락.
