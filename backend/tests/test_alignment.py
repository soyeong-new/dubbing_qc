from app.core.alignment import align, assign_scenes, group_by_scene
from app.schemas import SegmentText


def kr(s, e, t):
    return SegmentText(start=s, end=e, speaker="K", text=t)


def en(s, e, t):
    return SegmentText(start=s, end=e, speaker="E", text=t)


def test_align_matches_by_overlap():
    pairs = align(
        korean=[kr(1.0, 3.0, "밥 먹었어?"), kr(4.0, 6.0, "어이가 없네")],
        dubbed=[en(1.1, 3.2, "Did you eat rice?"), en(4.2, 6.1, "I have no kidney")],
    )
    assert len(pairs) == 2
    assert pairs[0].korean.text == "밥 먹었어?"
    assert pairs[0].dubbed.text == "Did you eat rice?"
    assert 0.8 < pairs[0].alignment_confidence <= 1.0


def test_align_reports_unmatched_korean():
    pairs = align(korean=[kr(1.0, 3.0, "대사"), kr(10.0, 12.0, "누락된 대사")],
                  dubbed=[en(1.0, 3.0, "line")])
    assert pairs[1].dubbed is None
    assert pairs[1].alignment_confidence == 0.0


def test_align_reports_extra_dubbed():
    pairs = align(korean=[kr(1.0, 3.0, "대사")],
                  dubbed=[en(1.0, 3.0, "line"), en(20.0, 22.0, "ad-lib")])
    extras = [p for p in pairs if p.korean is None]
    assert len(extras) == 1
    assert extras[0].dubbed.text == "ad-lib"


def test_assign_scenes_by_gap():
    pairs = align(
        korean=[kr(1.0, 2.0, "a"), kr(2.5, 4.0, "b"), kr(10.0, 11.0, "c")],
        dubbed=[en(1.0, 2.0, "a"), en(2.5, 4.0, "b"), en(10.0, 11.0, "c")],
    )
    pairs = assign_scenes(pairs, gap_threshold=3.0)
    assert pairs[0].scene_id == pairs[1].scene_id == "scene_1"
    assert pairs[2].scene_id == "scene_2"
    scenes = group_by_scene(pairs)
    assert len(scenes["scene_1"]) == 2


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
