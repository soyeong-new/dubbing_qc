import asyncio
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
            "우스꽝스럽게 들리는 문장이 있는가. 의미의 정확성은 평가하지 마십시오. "
            "추가로, 사전 필터에 걸리지 않은 애매한 민감 표현도 함께 확인하십시오: "
            "(4) 인종/성/종교/정치적으로 민감하거나 암시적으로 차별적인 표현이 있는가 "
            "(5) 등급(심의)에 영향을 줄 수 있는 수위의 욕설·폭력적 표현이 있는가. "
            "(1)~(3)에 해당하는 지적은 finding_type을 \"quality\"로, (4)~(5)에 "
            "해당하는 지적은 finding_type을 \"sensitive\"로 표기하십시오."
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
