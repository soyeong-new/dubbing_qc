from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

# v1 파이프라인에서는 사용하지 않지만, context.py/localization.py/voice_qc.py
# (플랜상 삭제하지 않고 보존하는 미연동 엔진 파일들)가 여전히 참조한다.
class ScriptSegment(BaseModel):
    id: str = Field(..., description="Segment ID")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker identifier")
    original_text: str = Field(..., description="Original Korean script line")
    translated_text: str = Field(..., description="English translated script line")


AXES = ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성", "억양 적합성"]


class SegmentText(BaseModel):
    start: float
    end: float
    speaker: str = "?"
    text: str


class AlignedPair(BaseModel):
    id: str
    korean: Optional[SegmentText] = None
    dubbed: Optional[SegmentText] = None
    scene_id: str = ""
    alignment_confidence: float = 1.0


class QCFinding(BaseModel):
    id: str
    segment_id: str
    category: str = Field(..., description="'localization' or 'voice'")
    severity: str = Field(..., description="'high' | 'medium' | 'low'")
    issue_type: str
    start_time: float
    end_time: float
    speaker: str
    description: str = Field(..., description="반드시 한국어")
    original_text: str
    current_translation: str
    recommendation: str = Field(..., description="반드시 영어 더빙 대사")
    confidence: float
    axis: str = "언어 적합성"
    source: str = "rule"
    agreement: int = 1
    alternatives: Dict[str, str] = Field(default_factory=dict)
    finding_type: Literal["quality", "sensitive"] = "quality"
    heard_korean: str = ""   # 모델이 실제로 들은 한국어 — 검수자가 청취 정확성을 검증하는 고리
    consensus: str = ""      # "2/2" 등 합의 수준. 빈 문자열 = 합의 미실시(룰 체크 등)


class HeldSegment(BaseModel):
    """판단 보류 구간 — 청취 불가/교차 불일치로 검증하지 못한 커버리지 공백."""
    scene_id: str
    segment_id: str = ""
    start: float
    end: float
    reason: str  # "청취 불가" | "교차 불일치"


class JudgeOutput(BaseModel):
    """페르소나 1회 호출의 결과: 지적 + 정직한 보류 목록."""
    findings: List["QCFinding"] = Field(default_factory=list)
    unheard_segment_ids: List[str] = Field(default_factory=list)


class AxisScore(BaseModel):
    axis: str
    mos: int = Field(..., ge=1, le=5)
    deduction_rate: float


class Verdict(BaseModel):
    status: Literal["pass", "conditional", "fail"]
    axis_scores: List[AxisScore]
    reasons: List[str] = Field(default_factory=list)


class QCJobInput(BaseModel):
    movie_title: str = "untitled"
    en_srt_path: str
    kr_srt_path: Optional[str] = None
    kr_audio_path: Optional[str] = None
    stem_audio_path: Optional[str] = None
    # QC 파이프라인은 사용하지 않는다 — 새로고침 후 화면을 복원할 때 영상
    # 미리보기를 다시 서빙하기 위해 job_id에 매달아 보관해 둘 뿐이다.
    original_media_path: Optional[str] = None
    dubbed_media_path: Optional[str] = None


class QCResult(BaseModel):
    verdict: Verdict
    findings: List[QCFinding]
    pairs: List[AlignedPair]
    held: List[HeldSegment] = Field(default_factory=list)


class FeedbackEntry(BaseModel):
    movie: str
    segment_id: str
    korean: str
    dubbed: str
    finding_id: str
    reviewer_action: Literal["approved", "rejected", "modified"]
    final_text: str = ""
    chosen_persona: str = ""
    timestamp: str = ""
