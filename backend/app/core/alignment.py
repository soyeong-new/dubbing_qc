from typing import Dict, List
from app.schemas import SegmentText, AlignedPair


def _overlap(a: SegmentText, b: SegmentText) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def align(korean: List[SegmentText], dubbed: List[SegmentText]) -> List[AlignedPair]:
    pairs: List[AlignedPair] = []
    used_dubbed = set()
    for i, kr in enumerate(korean):
        best_j, best_ov = None, 0.0
        for j, en in enumerate(dubbed):
            if j in used_dubbed:
                continue
            ov = _overlap(kr, en)
            if ov > best_ov:
                best_j, best_ov = j, ov
        if best_j is not None:
            en = dubbed[best_j]
            used_dubbed.add(best_j)
            union = max(kr.end, en.end) - min(kr.start, en.start)
            conf = round(best_ov / union, 3) if union > 0 else 0.0
            pairs.append(AlignedPair(id=f"pair_{i+1}", korean=kr, dubbed=en,
                                     alignment_confidence=conf))
        else:
            pairs.append(AlignedPair(id=f"pair_{i+1}", korean=kr, dubbed=None,
                                     alignment_confidence=0.0))
    for j, en in enumerate(dubbed):
        if j not in used_dubbed:
            pairs.append(AlignedPair(id=f"extra_{j+1}", korean=None, dubbed=en,
                                     alignment_confidence=0.0))
    pairs.sort(key=lambda p: (p.korean or p.dubbed).start)
    return pairs


def assign_scenes(pairs: List[AlignedPair], gap_threshold: float = 3.0) -> List[AlignedPair]:
    scene_num = 1
    prev_end = None
    for p in pairs:
        anchor = p.korean or p.dubbed
        if prev_end is not None and anchor.start - prev_end > gap_threshold:
            scene_num += 1
        p.scene_id = f"scene_{scene_num}"
        prev_end = max(prev_end or 0.0, anchor.end)
    return pairs


def group_by_scene(pairs: List[AlignedPair]) -> Dict[str, List[AlignedPair]]:
    scenes: Dict[str, List[AlignedPair]] = {}
    for p in pairs:
        scenes.setdefault(p.scene_id, []).append(p)
    return scenes
