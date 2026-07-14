# AETHER // AI Dubbing Quality Control Suite

AETHER는 한국 영화의 영어 AI 더빙 결과물을 검수(QC)하기 위한 사내 실무 도구입니다.
한국어 원본과 영어 더빙 완성본(SRT 자막 + 다이얼로그 스템 오디오)을 대조하여, 직역 오류·문화적
뉘앙스 손실·발화속도/싱크 문제·음질 이슈를 검출하고, 5축 MOS 스코어카드와 통과/조건부/반려
판정을 제공합니다.

설계 배경과 전체 아키텍처는 [`docs/superpowers/specs/2026-07-14-dubbing-qc-redesign-design.md`](docs/superpowers/specs/2026-07-14-dubbing-qc-redesign-design.md)를,
구현 계획은 [`docs/superpowers/plans/2026-07-14-dubbing-qc-pipeline.md`](docs/superpowers/plans/2026-07-14-dubbing-qc-pipeline.md)를 참고하세요.

---

## ⚙️ 주요 기능 (Key Features)

1. **파이프라인**: SRT 파싱(우선) + STT(폴백) → 한↔영 타임코드 정렬/씬 배정 → 결정론적 룰 체크
   (번역 누락, 발화속도, 싱크 오버플로, 클리핑/드롭아웃/잡음, SRT-음성 불일치) → 3-페르소나
   LLM Judge 패널 → 5축 MOS 판정.
2. **3-페르소나 Judge 패널**: 한국 문화·언어 전문가(원문 뉘앙스 보존), 영어 원어민 시청자(자연스러움),
   더빙 연출가(오디오 기반 감정·톤). 각 페르소나의 대안 수정안과 동의(agreement) 건수를 함께 제공합니다.
3. **5축 MOS 스코어카드**: 음질 / 감정 표현 / 싱크 정확도 / 자연스러움 / 언어 적합성.
   high 심각도 지적이 하나라도 있으면 점수와 무관하게 즉시 반려됩니다.
4. **검수 워크플로**: 리뷰어의 승인/반려(오탐)/직접수정 액션이 전부 서버에 기록되어(JSONL) 향후
   모델 학습 데이터로 축적됩니다. AI 판정은 가판정이며, 반려된 오탐을 제외한 "확정 재판정"을
   Report 탭에서 실행할 수 있습니다.
5. **3-탭 대시보드**: 프로젝트(업로드+진행률) / 검수(승인·반려·수정 + 페르소나 대안 비교) /
   판정·리포트(MOS 스코어카드 + 수정 지시서 + CSV/인쇄 내보내기).

---

## 🛠️ 기술 스택 (Technology Stack)

* **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, pytest
* **Frontend**: React 19, Vite, Vanilla CSS
* **AI Provider**: Gemini (`gemini-3.5-flash`, `google-generativeai`) — `backend/app/providers/`
  뒤에 추상화되어 있어 다른 모델로 교체 가능합니다.
* **오디오 처리**: ffmpeg (WAV 추출/파형/클립), 순수 Python 신호 분석(클리핑/드롭아웃/SNR)

### 보안 원칙: mock 자동 폴백 없음

`GEMINI_API_KEY`가 설정되지 않으면 QC 실행(`POST /api/qc/run`)은 **503으로 즉시 거부**됩니다.
목(mock) 데이터가 실제 검수 결과로 나오는 일은 없습니다. `MockProvider`는 `pytest` 실행 중에만
선택 가능하며, 운영 코드 경로에는 폴백 분기 자체가 존재하지 않습니다.

---

## 🚀 실행 방법 (Quick Start)

### 1단계: API 키 설정 (필수)

```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

키가 없으면 QC 실행이 거부됩니다(위 보안 원칙 참고).

### 2단계: 서버 동시 구동

```bash
./start.sh
```

* **프론트엔드 대시보드 URL**: [http://localhost:5173](http://localhost:5173)
* **백엔드 API 문서**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3단계: 백엔드 테스트 실행

```bash
cd backend
venv/bin/python -m pytest -q
```

---

## 📂 프로젝트 구조 (Structure)

```text
dubbing_qc/
├── start.sh                    # 백엔드 & 프론트엔드 동시 실행 쉘 스크립트
├── README.md                   # 가이드 문서
├── docs/superpowers/           # 설계 스펙 + 구현 계획
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI: 잡 기반 QC 실행/진행률/피드백/재판정/CSV
│   │   ├── schemas.py          # Pydantic 데이터 계약 (AlignedPair, QCFinding, Verdict 등)
│   │   ├── qc_config.yaml      # 판정 임계값·감점 가중치 설정
│   │   ├── core/
│   │   │   ├── pipeline.py     # 오케스트레이션: ingest→정렬→룰체크→패널→판정
│   │   │   ├── ingest.py       # SRT 파싱(우선) / STT(폴백)
│   │   │   ├── alignment.py    # 한↔영 타임코드 정렬 + 씬 배정
│   │   │   ├── rule_checks.py  # 결정론적 체크 (텍스트 + 오디오 신호)
│   │   │   ├── judge_panel.py  # 3-페르소나 LLM Judge + 병합기
│   │   │   ├── verdict.py      # 5축 MOS 판정 엔진
│   │   │   ├── context.py      # (미연동) 시각 맥락 레이어 — 향후 실모델 연동용
│   │   │   ├── localization.py # (미연동) 초기 프로토타입 텍스트 분석기
│   │   │   └── voice_qc.py     # (미연동) 초기 프로토타입 음성 분석기
│   │   ├── providers/          # 모델 프로바이더 추상화 (Gemini/Mock, 교체 가능)
│   │   ├── knowledge/          # 호칭/관용구 지식베이스 (YAML, LLM 프롬프트 주입용)
│   │   └── feedback/           # 검수 피드백 JSONL 저장소
│   ├── tests/                  # pytest 스위트
│   └── requirements.txt
│
└── frontend/
    ├── index.html
    └── src/
        ├── App.jsx              # 탭 전환 + 검수(Review) 뷰
        ├── App.css
        ├── api.js               # 백엔드 API 클라이언트
        └── views/
            ├── ProjectView.jsx  # 업로드 + 진행률
            └── ReportView.jsx   # MOS 스코어카드 + 판정 + 내보내기
```

`core/context.py`, `core/localization.py`, `core/voice_qc.py`는 초기 프로토타입에서 넘어온
파일로, 현재 파이프라인(`pipeline.py`)이 호출하지 않는 미연동 상태로 보존되어 있습니다.
