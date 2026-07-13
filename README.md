# AETHER // AI Dubbing Quality Control Suite

AETHER는 한국 영화의 영어 AI 더빙 결과물을 자동으로 검수(QC)하기 위해 개발된 인공지능 기반 품질 관리 플랫폼입니다.
시각적/상황 맥락(Scene Context)을 학습한 AI가 번역 스크립트와 더빙 음성 데이터를 복합 분석하여 품질 점수를 산출하고 수정안을 추천합니다.

---

## ⚙️ 주요 기능 (Key Features)

1. **Visual & Audio Monitoring**: 비디오 재생 시 실시간으로 오디오 파형(Waveform)이 연동 및 가시화되며, 현재 대사와 자막이 싱크에 맞게 실시간 동기화됩니다.
2. **Context-Aware Localization QC**:
   - **호칭 오류 감지**: 친근하게 부르는 '형/오빠/누나' 등이 문맥에 맞지 않게 'brother/sister'로 직역된 오류를 검출합니다.
   - **문화적 뉘앙스/관용구 검출**: '눈치 보다', '어이가 없네' 등이 지나치게 직역된 부자연스러운 영문 번역을 자동으로 탐지합니다.
3. **Voice & Sync QC Engine**:
   - **Pacing & Sync Overflow**: 한정된 타임코드 대비 글자 수/음절 수가 지나치게 길어 배우가 말이 매우 빨라져 입 모양 싱크가 어긋나는 지점을 파악합니다.
   - **음색 일관성(Consistency)**: 이전 씬에서 정립된 화자의 음색 지문(Timbre embedding)과 비교해 톤앤매너나 성우의 음색이 변경된 구간을 검출합니다.
4. **Interactive QC Dashboard**: 검수자가 AI의 리스크 위험 등급(High/Medium/Low) 분류 카드를 확인하고, **"Apply AI Fix"** 버튼 클릭 한 번으로 수정을 즉시 적용할 수 있습니다.

---

## 🛠️ 기술 스택 (Technology Stack)

* **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic
* **Frontend**: React 19, Vite, Vanilla CSS (Outfit Font / JetBrains Mono)
* **AI Model Engine (Stubs & Integrations)**:
  - **Gemini 1.5 Flash API**: 다국어 영상 맥락 분석 및 다이어로그 오역 평가
  - **WhisperX (Forced Alignment)** 및 **PyAnnote (Diarization)**: 싱크 검사 및 음색 검사의 실제 코어 모델 추천

---

## 🚀 실행 방법 (Quick Start)

### 1단계: API 키 설정 (선택 사항)
실제 Gemini AI 모델을 연동하여 실시간 번역 검수를 돌리고 싶다면, 아래와 같이 환경변수를 설정합니다:
```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```
*(API 키가 설정되지 않은 경우, 자동으로 시스템 내장 고정밀 Mock 검수 엔진이 작동하여 문제없이 테스트하실 수 있습니다!)*

### 2단계: 서버 동시 구동
프로젝트 루트 디렉토리에서 동시 구동 스크립트를 실행합니다:
```bash
./start.sh
```
* **프론트엔드 대시보드 URL**: [http://localhost:5173](http://localhost:5173)
* **백엔드 API 문서**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 프로젝트 구조 (Structure)

```text
dubbing_qc/
├── start.sh                 # 백엔드 & 프론트엔드 동시 실행 쉘 스크립트
├── README.md                # 가이드 문서
│
├── backend/                 # FastAPI 백엔드 파이프라인
│   ├── app/
│   │   ├── main.py          # FastAPI 진입점 및 Mock 데이터 로더
│   │   ├── schemas.py       # Pydantic 데이터 스키마
│   │   └── core/
│   │       ├── pipeline.py  # QC 검수 총괄 관리자
│   │       ├── context.py   # 시각 및 맥락 레이어
│   │       ├── localization.py # LLM 연동 텍스트 오역 분석기
│   │       └── voice_qc.py  # 음성/싱크 검출 분석기
│   └── requirements.txt
│
└── frontend/                # Vite + React 대시보드 웹앱
    ├── index.html           # SEO 및 폰트 설정
    └── src/
        ├── App.jsx          # 인터랙티브 대시보드 핵심 로직
        ├── App.css          # 프리미엄 다크테마 CSS
        └── index.css        # 브라우저 초기화 CSS
```
