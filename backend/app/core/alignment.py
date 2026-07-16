from typing import Dict, List
from app.schemas import SegmentText, AlignedPair


def _overlap(a: SegmentText, b: SegmentText) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def align(korean: List[SegmentText], dubbed: List[SegmentText]) -> List[AlignedPair]:
    """더빙(영어) 자막 한 줄마다, 시간이 겹치는 한국어 세그먼트를 전부 찾아 붙인다.

    한국어 SRT와 영어 SRT처럼 양쪽 줄 수가 비슷할 때는 대부분 1:1로 매칭된다.
    로컬 STT처럼 한국어가 훨씬 큰 단위(청크)로 묶여 나올 때는, 그 청크 하나가
    여러 영어 줄에 걸쳐 있을 수 있다 — 이 경우 겹치는 모든 영어 줄이 그 한국어
    청크를 공유해서 받는다(하나만 골라 배정하고 나머지를 비우지 않는다). 실제로
    원본에서도 그 시간 동안 쭉 이어 말한 것이므로, 여러 줄이 같은 원문을 공유하는
    것이 사실에 부합한다.
    """
    pairs: List[AlignedPair] = []
    matched_korean = set()
    for j, en in enumerate(dubbed):
        overlapping = [(i, kr) for i, kr in enumerate(korean) if _overlap(kr, en) > 0]
        if overlapping:
            for i, _ in overlapping:
                matched_korean.add(i)
            texts = [kr.text for _, kr in overlapping]
            starts = [kr.start for _, kr in overlapping]
            ends = [kr.end for _, kr in overlapping]
            merged_kr = SegmentText(
                start=min(starts), end=max(ends),
                speaker=overlapping[0][1].speaker, text=" ".join(texts),
            )
            best_ov = max(_overlap(kr, en) for _, kr in overlapping)
            union = max(merged_kr.end, en.end) - min(merged_kr.start, en.start)
            conf = round(best_ov / union, 3) if union > 0 else 0.0
            pairs.append(AlignedPair(id=f"pair_{j+1}", korean=merged_kr, dubbed=en,
                                     alignment_confidence=conf))
        else:
            pairs.append(AlignedPair(id=f"pair_{j+1}", korean=None, dubbed=en,
                                     alignment_confidence=0.0))
    for i, kr in enumerate(korean):
        if i not in matched_korean:
            pairs.append(AlignedPair(id=f"extra_{i+1}", korean=kr, dubbed=None,
                                     alignment_confidence=0.0))
    pairs.sort(key=lambda p: (p.korean or p.dubbed).start)
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
