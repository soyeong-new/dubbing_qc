# 로컬 STT 도입 + 업로드 경량화 설계

## 1. 배경

v2 오디오 확장 작업을 실사용 영화(약 2시간짜리 장편)에 적용하는 과정에서 두 가지 실제 운영 장애가 발견됐다.

**문제 A — 원본 한국어 오디오 STT가 대부분 누락됨.** 한국어 SRT가 없을 때 `ingest.py`의
`load_text_source()`는 원본 오디오 전체를 Gemini에 한 번의 요청으로 보내 STT를 요청한다.
실제 119분짜리 영화로 확인한 결과, Gemini 응답은 앞부분 약 17분(142개 세그먼트)만 반환하고
나머지는 응답 길이 한계로 잘렸다. 결과적으로 2168개 페어 중 2026개가 `korean: null`이 되어
`번역 누락` 등 오탐이 대량 발생했다.

**문제 B — 자막-음성 대조 체크가 API 할당량에 막혀 죽음.** `rule_checks.py`의
`check_srt_audio_match`는 더빙 오디오를 샘플링해 Gemini STT로 재전사하고 자막 텍스트와
유사도를 비교한다. 실제로는 이 비교 자체가 3-페르소나 패널(특히 오디오를 직접 듣는 원어민/
연출가)이 이미 하는 일과 중복이며, 원래 의도는 "내용이 같은가"가 아니라 "자막이 주장하는
구간에 실제로 발화가 시작/종료되는가"였다. 현재 구현은 이 의도와 다르게 STT+텍스트 유사도
비교로 만들어져 있었고, Gemini 무료 티어 일일 호출 한도(20회)를 순식간에 소진해 대부분의
샘플에서 429 오류로 실패한다.

이 설계는 두 문제를 근본적으로 해결하고, 겸사겸사 발견된 업로드 경로의 낭비(대용량 영상
파일이 오디오만 필요한데도 그대로 서버에 전송되고, 검수 탭이 같은 파일을 중복 업로드하는
문제)도 함께 정리한다.

## 2. 스코프

**포함:**
- 한국어 원본 오디오 STT를 로컬 모델(Whisper 파인튜닝 모델)로 전환
- `check_srt_audio_match`를 STT 없는 순수 신호처리 기반 발화 타이밍 동기화 체크로 재설계
- `ModelProvider` 인터페이스에서 `transcribe()` 완전 제거 (더 이상 쓰는 곳이 없음)
- 프론트엔드: 영상 업로드 전 오디오만 추출해서 전송(`original`, `dubbed` 역할)
- 프론트엔드: 검수 탭의 중복 업로드 로직 제거, Project 탭에서 받은 데이터 재사용

**제외 (범위 밖):**
- 영어 STT 폴백 개선 — `en_srt_path`가 스키마상 필수라 실제로 트리거되지 않는 경로이므로
  건드리지 않는다.
- 화자 분리(diarization) — `batiai/batispeak-diarize` 모델이 존재하지만, 이번 스코프는
  STT 텍스트 확보가 목적이라 화자 라벨은 기존과 동일하게 기본값(`"?"`)으로 둔다.
- 배포 아키텍처(서버리스 대응 등) — 기존 v2 설계 문서에서 이미 별도 미해결 질문으로
  다루고 있어 이번 설계에서 재론하지 않는다.

## 3. 컴포넌트별 설계

### 3.1 로컬 한국어 STT (`backend/app/core/local_stt.py`, 신규)

- 모델: `batiai/batisay-ko-turbo` (Whisper Large-v3를 한국어로 파인튜닝, SafeTensors 포맷,
  `transformers` 라이브러리로 로드 — 이미 설치된 `torch` 재사용)
- 긴 오디오(2시간) 처리는 `transformers`의 automatic-speech-recognition 파이프라인이 내장
  지원하는 청킹(`chunk_length_s`/`stride_length_s`)과 세그먼트 타임스탬프 반환 기능을
  그대로 사용한다 — 직접 파일을 잘라 여러 번 호출하는 로직은 만들지 않는다. (이 청킹은
  Gemini의 응답 길이 제한을 우회하기 위한 것이 아니라, Whisper 계열 모델이 원래 고정 길이
  창(30초)으로 오디오를 처리하는 구조적 특성 때문이며, 완전히 다른 레이어에서 일어나는
  씬 배치(`assign_scenes`, 판정 호출 그룹핑)와는 무관하다.)
- Apple Silicon Mac 기준 `device="mps"`(GPU 가속) 우선 시도, 실패 시 `device="cpu"`로 폴백
- 실제 모델은 함수 내부에서 지연 로드(`_get_model()` 캐시), 자동화 테스트는 이 실제 모델을
  절대 로드하지 않는다 — `accent.py`가 이미 확립한 패턴을 그대로 따른다.

인터페이스:
```python
def transcribe_korean(
    audio_path: str,
    transcribe_fn: Optional[Callable[[str], list]] = None,
) -> List[SegmentText]:
    ...
```
`transcribe_fn`은 실제 모델 호출부를 감싼 내부 헬퍼를 대체할 수 있는 주입 지점이며,
테스트에서는 가짜 함수(고정된 세그먼트 리스트 반환)를 넣어 실제 모델 로드를 완전히 피한다.

### 3.2 `ingest.py` 연동

`load_text_source(lang, srt_path, audio_path)`에서 `provider` 파라미터를 제거한다.
`lang == "ko"`이고 `srt_path`가 없으면 `local_stt.transcribe_korean(audio_path)`를 호출한다.
`lang == "en"` 경로는 이번 스코프에서 다루지 않지만(§2 제외 항목), 함수 시그니처에서
`provider`가 빠지므로 호출부(`pipeline.py`)도 함께 정리한다.

### 3.3 `check_srt_audio_match` → `check_dialogue_timing_sync` 재설계

**의도 재정의:** "더빙 오디오의 내용이 자막과 같은가"가 아니라 "원본과 더빙, 두 오디오
트랙에서 실제로 같은 시간대에 발화가 시작·종료되는가"를 확인한다. 내용 일치는 이미
3-페르소나 패널이 오디오를 직접 들으며 판단하므로 중복 검증하지 않는다.

**알고리즘:**
1. 정렬된 각 페어에 대해 원본 오디오(`kr_audio_path`)에서 `korean.start~end` 구간,
   더빙 스템(`stem_audio_path`)에서 `dubbed.start~end` 구간을 각각 추출한다.
2. `check_audio_quality`가 이미 사용하는 RMS 에너지 기반 무음 판정 로직을 재사용해
   각 구간 내에서 실제 발화가 시작되는 지점(첫 비무음 프레임)과 끝나는 지점(마지막
   비무음 프레임)을 찾는다.
3. 두 트랙의 발화 시작 지점 차이가 허용 오차(0.5초, `check_sync_overflow`와 동일 기준)를
   넘으면 지적을 생성한다.
4. STT도 LLM도 호출하지 않는다 — 순수 신호처리이므로 API 비용도, 할당량 제한도 없다.

이 재설계로 기존에 있던 `provider.transcribe()` 실패 시 재시도 로직(커밋 `e7e0f32`에서
프로덕션 긴급 조치로 추가됨)은 자연스럽게 불필요해진다. 그 조치는 당시 아키텍처
기준으로는 올바른 임시 조치였고, 이번 재설계로 대체된다.

### 3.4 `ModelProvider` 인터페이스 정리

`transcribe()`를 호출하는 곳은 코드베이스 전체에서 `ingest.py`와
`rule_checks.py::check_srt_audio_match` 두 곳뿐이었다(전수 검색으로 확인). 두 곳 모두
§3.2, §3.3으로 대체되므로 `transcribe()`는 어디서도 호출되지 않게 된다.

- `backend/app/providers/base.py` — `ModelProvider.transcribe()` 추상 메서드 삭제
- `backend/app/providers/gemini.py` — `GeminiProvider.transcribe()` 구현 삭제
- `backend/app/providers/mock.py` — `MockProvider.transcribe()` 구현 삭제
- 관련 테스트(`test_providers.py`의 transcribe 테스트, 각 테스트 파일의 `transcribe` 스텁)
  정리. `test_api.py`의 `/api/qc/transcribe` 404 확인 테스트는 애초에 그런 엔드포인트가
  없다는 걸 확인하는 것이라 이번 변경과 무관하게 유지한다.

### 3.5 프론트엔드 — 업로드 전 오디오 추출

`original`(한국어 원본), `dubbed`(영어 더빙 완성본) 두 업로드 슬롯은 현재 영상 파일
전체(수 GB)를 서버로 전송하지만, 서버는 오디오 트랙만 필요로 한다(ffmpeg로 변환 후
audio_path만 사용). 영상 미리보기는 이미 `URL.createObjectURL(file)`로 로컬에서만
재생되고 있어 서버 전송과 무관하다.

- `ffmpeg.wasm`을 도입해 브라우저에서 업로드 직전에 오디오 트랙만 추출한다.
- 서버로는 추출된 오디오 파일만 전송한다 (기존 영상 업로드 대비 전송량이 크게 줄어듦 —
  실측 사례 기준 약 3GB 영상 → 약 228MB 오디오 수준).
- Vite 개발 서버에 `ffmpeg.wasm`이 요구하는 크로스 오리진 격리 헤더(COOP/COEP)를
  추가한다.
- `stem` 슬롯은 이미 오디오 전용이라 변경하지 않는다.

### 3.6 검수 탭 업로드 중복 제거

`App.jsx`에 `ProjectView.jsx`와 별개로 동작하는 업로드 로직(`handleOriginalUpload`,
`handleVideoUpload`, "미디어 등록" 버튼)이 있었다. 이는 검수 탭에서 영상 재생과 파형
표시를 위해 같은 파일을 서버에 다시 업로드하는데, `uploadMedia()` 응답에 이미
`audio_path`와 `waveform`이 담겨 있어 Project 탭의 최초 업로드 결과(`uploads` 상태)만
재사용해도 충분하다.

- `handleOriginalUpload`/`handleVideoUpload`와 관련 UI(버튼, input) 제거
- 검수 탭의 영상 재생/파형 렌더링은 `uploads.original`/`uploads.dubbed`(Project 탭에서
  이미 채워진 상태)를 직접 참조하도록 변경
- 로컬 영상 미리보기용 `File` 객체는 Project 탭 업로드 시점에 `uploads` 상태에 함께
  저장해 검수 탭에서 재사용

## 4. 테스트 전략

- `local_stt.py`: 실제 모델을 자동화 테스트에서 절대 로드하지 않는다. `transcribe_fn`
  주입으로 실제 모델 호출부를 대체.
- `check_dialogue_timing_sync`: `check_audio_quality`와 동일하게 순수 신호처리이므로
  synthetic wav 파일(sine wave + 무음 구간)로 직접 테스트 가능, 모델/네트워크 불필요.
- 프론트엔드는 기존과 동일하게 자동화 테스트 스위트가 없으므로, `npm run build` 성공과
  수동 확인으로 검증한다.

## 5. 자체 검토

- **플레이스홀더 스캔:** 없음 — 모든 섹션이 구체적인 파일 경로와 동작을 명시함.
- **내부 일관성:** §3.3에서 삭제되는 재시도 로직(커밋 `e7e0f32`)과 §3.4에서 삭제되는
  `transcribe()` 인터페이스가 서로 맞물려 있음을 명시적으로 기술함 — 순서상 §3.3, §3.4는
  같은 계획의 연속된 태스크로 묶여야 함.
- **스코프 점검:** 6개 컴포넌트 모두 "Gemini STT 의존성 제거 + 그 과정에서 드러난 업로드
  낭비 정리"라는 단일 주제로 수렴하므로 하나의 구현 계획으로 다루기에 적절한 범위다.
- **모호성 점검:** "허용 오차 0.5초"는 기존 `check_sync_overflow`와 동일한 값을 명시적으로
  채택했다. 모델 디바이스 선택(`mps` 우선, `cpu` 폴백)도 명시적으로 정했다.

## 6. 미해결 질문 (구현 계획에서 다루지 않음, 추후 필요 시 별도 논의)

- `batiai/batisay-ko-turbo` 모델 다운로드 크기·라이선스 조건 확인 필요 (실제 다운로드는
  구현 단계에서 최초 1회 발생)
- `ffmpeg.wasm`의 브라우저 호환성(구형 브라우저에서 SharedArrayBuffer 미지원 시 폴백 없음)
