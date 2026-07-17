from typing import Dict, List
from app.schemas import SegmentText, AlignedPair


def _overlap(a: SegmentText, b: SegmentText) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def align(korean: List[SegmentText], dubbed: List[SegmentText],
         min_overlap_ratio: float = 0.5) -> List[AlignedPair]:
    """더빙(영어) 자막 한 줄마다, 그 줄과 "충분히" 겹치는 한국어 세그먼트를 찾아 붙인다.

    한국어 SRT와 영어 SRT처럼 양쪽 줄 수가 비슷할 때는 대부분 1:1로 매칭된다.
    로컬 STT처럼 한국어 문장 하나가 여러 영어 줄에 걸쳐 있을 때는, 겹치는 여러
    영어 줄이 그 한국어 문장을 공유해서 받는다 — 실제로 원본에서 그 시간 동안 쭉
    이어 말한 것이므로 여러 줄이 같은 원문을 공유하는 것이 사실에 부합한다.

    "충분히" 겹친다는 기준은 두 세그먼트 중 짧은 쪽 길이 대비 겹친 시간의 비율이
    min_overlap_ratio(기본 50%) 이상인 경우다. 단순히 "조금이라도 겹치면 포함"하면
    문장 경계가 영어 줄 경계와 정확히 일치하지 않아 이웃 줄과 아주 살짝(0.1~0.5초)만
    겹치는 흔한 경우까지 전부 끌려와, 상관없는 영어 줄에 엉뚱하게 긴 한국어가
    붙어버린다(실측 확인). 비율 기준을 두면 이런 경계의 사소한 겹침은 걸러내면서도,
    짧은 영어 줄 여러 개가 긴 한국어 발화 하나에 온전히 포함되는 정당한 공유
    케이스는 그대로 유지된다(포함되는 쪽 길이 전체가 겹치므로 비율이 100%에 가깝다).

    영어 자막 줄 어디와도 겹치지 않는 한국어 세그먼트는 결과에서 제외한다 —
    영어 대사가 존재하지 않는 시간대(효과음, 비명, 배경 소음 등)에서 로컬 STT가
    주워듣거나 환각으로 지어낸 단어가 실측으로 다수 확인되었고, 이런 단어들이
    "번역 누락"으로 오탐되어 검수 결과를 어지럽혔다. 영어 SRT를 기준(주체)으로
    삼는다는 원칙에 따라, 대응하는 영어 대사가 없는 구간은 애초에 검수 대상이
    아니다.
    """
    pairs: List[AlignedPair] = []
    for j, en in enumerate(dubbed):
        overlapping = []
        for kr in korean:
            ov = _overlap(kr, en)
            if ov <= 0:
                continue
            shorter = min(kr.end - kr.start, en.end - en.start)
            if shorter > 0 and ov / shorter >= min_overlap_ratio:
                overlapping.append(kr)
        if overlapping:
            overlapping.sort(key=lambda kr: kr.start)
            texts = [kr.text for kr in overlapping]
            starts = [kr.start for kr in overlapping]
            ends = [kr.end for kr in overlapping]
            merged_kr = SegmentText(
                start=min(starts), end=max(ends),
                speaker=overlapping[0].speaker, text=" ".join(texts),
            )
            best_ov = max(_overlap(kr, en) for kr in overlapping)
            union = max(merged_kr.end, en.end) - min(merged_kr.start, en.start)
            conf = round(best_ov / union, 3) if union > 0 else 0.0
            pairs.append(AlignedPair(id=f"pair_{j+1}", korean=merged_kr, dubbed=en,
                                     alignment_confidence=conf))
        else:
            pairs.append(AlignedPair(id=f"pair_{j+1}", korean=None, dubbed=en,
                                     alignment_confidence=0.0))
    pairs.sort(key=lambda p: p.dubbed.start)
    return pairs


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


def group_by_scene(pairs: List[AlignedPair]) -> Dict[str, List[AlignedPair]]:
    scenes: Dict[str, List[AlignedPair]] = {}
    for p in pairs:
        scenes.setdefault(p.scene_id, []).append(p)
    return scenes
