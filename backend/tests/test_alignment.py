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


def test_align_drops_korean_with_no_matching_dubbed_line():
    # 영어 SRT가 기준(주체)이다 — 대응하는 영어 대사가 없는 시간대의 한국어는
    # 결과에서 아예 제외한다. 이런 구간(효과음·비명 등)에서 로컬 STT가 주워듣거나
    # 환각으로 지어낸 단어가 실측으로 다수 확인되었고, 전부 "번역 누락"으로
    # 오탐되어 검수 결과를 어지럽혔기 때문이다.
    pairs = align(korean=[kr(1.0, 3.0, "대사"), kr(10.0, 12.0, "영어 자막이 없는 구간의 소리")],
                  dubbed=[en(1.0, 3.0, "line")])
    assert len(pairs) == 1
    assert pairs[0].korean.text == "대사"


def test_align_reports_extra_dubbed():
    pairs = align(korean=[kr(1.0, 3.0, "대사")],
                  dubbed=[en(1.0, 3.0, "line"), en(20.0, 22.0, "ad-lib")])
    extras = [p for p in pairs if p.korean is None]
    assert len(extras) == 1
    assert extras[0].dubbed.text == "ad-lib"


def test_align_shares_one_korean_chunk_across_multiple_dubbed_lines():
    # 로컬 STT처럼 한국어가 훨씬 큰 청크로 묶여 나오면, 그 청크 하나가 걸쳐 있는
    # 여러 영어 줄이 전부 그 한국어 텍스트를 공유해서 받아야 한다 — 하나만 받고
    # 나머지가 비면 안 된다.
    pairs = align(
        korean=[kr(0.0, 10.0, "쭉 이어 말한 원본 대사")],
        dubbed=[en(0.0, 2.0, "Line one"), en(2.0, 5.0, "Line two"), en(5.0, 9.0, "Line three")],
    )
    assert len(pairs) == 3
    assert all(p.korean is not None for p in pairs)
    assert all(p.korean.text == "쭉 이어 말한 원본 대사" for p in pairs)
    assert [p.dubbed.text for p in pairs] == ["Line one", "Line two", "Line three"]


def test_align_excludes_marginal_boundary_overlap_but_keeps_substantial_overlap():
    # 실사용에서 발견된 문제: 문장 경계가 영어 줄 경계와 정확히 일치하지 않아
    # 이웃 한국어 문장이 영어 줄과 아주 살짝(여기서는 0.5초, 짧은 쪽 길이의 33%)만
    # 겹치는 경우, 그 문장까지 통째로 끌려와 상관없는 영어 줄에 엉뚱하게 긴
    # 한국어가 붙어버렸다(실측 확인). 겹침 비율이 낮으면 제외되어야 한다.
    sentence_a = kr(150.7, 153.0, "자기야 난 정말 자기 사랑하는거 알지?")  # 영어 줄과 72% 겹침
    sentence_b = kr(153.0, 154.5, "어 자기야 자기.")  # 영어 줄과 33%만 겹침 (경계)
    en_line = en(151.7, 153.5, "I don't like this place. Let's.")
    next_en_line = en(153.6, 154.6, "hey, Pay up or else.")

    pairs = align(korean=[sentence_a, sentence_b], dubbed=[en_line, next_en_line])

    line_pair = next(p for p in pairs if p.dubbed.text == en_line.text)
    assert line_pair.korean.text == "자기야 난 정말 자기 사랑하는거 알지?"
    assert "자기야 자기" not in line_pair.korean.text  # 경계에 살짝 겹친 문장은 제외


def test_align_splits_unpunctuated_korean_words_across_separate_english_lines():
    # 실사용에서 발견된 문제: Whisper가 세 개의 별개 발화("여자가 미쳤나 보네" /
    # "자기야 우리 그냥 여기서 나가자" / "자기야 나 정말 사랑하는 거 맞아?") 사이에
    # 문장부호를 안 찍어서, 미리 문장 단위로 묶으면 이 셋이 하나의 덩어리가 되어
    # 서로 다른 영어 줄 세 개에 전부 똑같이 통째로 붙어버렸다. 해결책은 문장으로
    # 미리 묶지 않고 단어 그대로 align에 넘기는 것 — 각 영어 줄은 자신의 시간
    # 구간에 실제로 겹치는 단어들만 시간순으로 받아야 한다(어순은 항상 한국어가
    # 말해진 시간순이므로 영어 어순과 무관하게 보존된다).
    words = [
        kr(146.4, 146.8, "여자가"), kr(146.8, 147.2, "미쳤나"), kr(147.2, 147.6, "보네"),
        kr(147.9, 148.2, "자기야"), kr(148.2, 148.5, "우리"), kr(148.5, 148.8, "그냥"),
        kr(148.8, 149.0, "여기서"), kr(149.0, 149.1, "나가자"),
        kr(149.3, 149.6, "자기야"), kr(149.6, 149.9, "나"), kr(149.9, 150.2, "정말"),
        kr(150.2, 150.5, "사랑하는"), kr(150.5, 150.6, "거"), kr(150.6, 150.7, "맞아?"),
    ]
    dubbed = [
        en(146.4, 147.7, "This woman's crazy!"),
        en(147.9, 149.1, "Let's get out of here!"),
        en(149.3, 150.7, "Honey, do you even love me?!"),
    ]

    pairs = align(korean=words, dubbed=dubbed)

    line1 = next(p for p in pairs if p.dubbed.text == "This woman's crazy!")
    line2 = next(p for p in pairs if p.dubbed.text == "Let's get out of here!")
    line3 = next(p for p in pairs if p.dubbed.text == "Honey, do you even love me?!")
    assert line1.korean.text == "여자가 미쳤나 보네"
    assert line2.korean.text == "자기야 우리 그냥 여기서 나가자"
    assert line3.korean.text == "자기야 나 정말 사랑하는 거 맞아?"


def test_align_merges_multiple_korean_segments_overlapping_one_dubbed_line():
    # 반대로 짧은 한국어 세그먼트 여러 개가 영어 자막 한 줄에 걸쳐 있으면
    # 텍스트를 이어붙여서 하나로 합친다.
    pairs = align(
        korean=[kr(0.0, 1.0, "첫마디"), kr(1.0, 2.0, "둘째마디")],
        dubbed=[en(0.0, 2.0, "Combined line")],
    )
    assert len(pairs) == 1
    assert pairs[0].korean.text == "첫마디 둘째마디"


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
