from typing import Callable, List, Optional
from app.schemas import AlignedPair, QCFinding

MODEL_SOURCE = "Jzuluaga/accent-id-commonaccent_ecapa"  # 실제 모델 ID는 도입 시 재검증
MODEL_SAVEDIR = "/tmp/aether_accent_model"

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    from speechbrain.inference.classifiers import EncoderClassifier
    _classifier = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=MODEL_SAVEDIR)
    return _classifier


def classify_accent(wav_path: str) -> tuple:
    """단일 오디오 클립의 억양을 분류한다. (라벨, 확신도)를 반환한다.

    실제 SpeechBrain 모델을 로드하므로 자동화 테스트에서는 호출하지 않는다 —
    check_accent_conformance()의 classify_fn 주입으로 대체한다.
    """
    classifier = _get_classifier()
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    return text_lab[0], float(score[0])


def check_accent_conformance(
    pairs: List[AlignedPair], stem_wav_path: str,
    extract_clip_fn: Optional[Callable] = None,
    classify_fn: Optional[Callable] = None,
    target_accent: str = "us",
    confidence_threshold: float = 0.6,
) -> List[QCFinding]:
    from app.core.rule_checks import extract_clip as default_extract_clip
    extract_clip_fn = extract_clip_fn or default_extract_clip
    classify_fn = classify_fn or classify_accent

    findings = []
    for p in pairs:
        if not p.dubbed or not p.dubbed.text.strip():
            continue
        clip = extract_clip_fn(stem_wav_path, p.dubbed.start, p.dubbed.end)
        label, confidence = classify_fn(clip)
        if label.lower() != target_accent or confidence < confidence_threshold:
            findings.append(QCFinding(
                id=f"accent_{p.id}", segment_id=p.id, category="voice",
                severity="medium", issue_type="억양 부적합",
                start_time=p.dubbed.start, end_time=p.dubbed.end, speaker=p.dubbed.speaker,
                description=f"이 세그먼트의 억양이 목표 표준과 다르게 분류되었습니다 "
                            f"(분류: {label}, 확신도 {confidence:.2f}).",
                original_text=p.korean.text if p.korean else "",
                current_translation=p.dubbed.text,
                recommendation="Re-record with the target accent or review voice casting.",
                confidence=confidence, axis="억양 적합성", source="rule",
            ))
    return findings
