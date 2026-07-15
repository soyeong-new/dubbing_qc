import asyncio
from typing import Callable, Optional
from app.schemas import QCJobInput, QCResult
from app.providers.base import ModelProvider, get_provider
from app.core.ingest import load_text_source
from app.core.alignment import align, assign_scenes, group_by_scene
from app.core.rule_checks import (
    run_text_checks, check_audio_quality, check_srt_audio_match, check_sensitive_words,
)
from app.core.accent import check_accent_conformance
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

        # ④ 페르소나 패널 (연출가에게 원본 오디오도 함께 전달)
        scenes = group_by_scene(pairs)
        panel_findings = await run_panel(
            scenes, load_knowledge(), provider,
            stem_wav_path=job.stem_audio_path,
            kr_audio_path=job.kr_audio_path,
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
