from typing import List, Optional
from app.providers.base import ModelProvider, Persona
from app.schemas import AlignedPair, QCFinding

# 결정론적 테스트 더블. 운영 경로에서는 base.get_provider()가 선택을 차단한다.
_BAD_PATTERNS = [
    ("kidney", "번역 오류", "high", "관용구 '어이가 없네'가 신장(kidney)으로 오역되었습니다.", "This is ridiculous."),
    ("eat rice", "문화적 정서 차이", "medium", "'밥 먹었어?'가 직역되어 안부 인사의 의미가 사라졌습니다.", "Have you eaten?"),
    ("brother", "문화적 정서 차이", "medium", "호칭 '형'이 brother로 직역되어 어색합니다.", "Hey, man."),
]


class MockProvider(ModelProvider):
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        findings = []
        for pair in pairs:
            if not pair.dubbed or not pair.korean:
                continue
            text_en = pair.dubbed.text.lower()
            for pattern, issue_type, severity, desc, rec in _BAD_PATTERNS:
                if pattern in text_en:
                    findings.append(QCFinding(
                        id=f"{persona.key}_{pair.id}_{pattern.replace(' ', '_')}",
                        segment_id=pair.id, category="localization",
                        severity=severity, issue_type=issue_type,
                        start_time=pair.korean.start, end_time=pair.korean.end,
                        speaker=pair.korean.speaker, description=desc,
                        original_text=pair.korean.text,
                        current_translation=pair.dubbed.text,
                        recommendation=rec, confidence=0.9,
                        axis=persona.axes[0] if persona.axes else "언어 적합성",
                        source=f"persona:{persona.key}",
                    ))
                    break
        return findings
