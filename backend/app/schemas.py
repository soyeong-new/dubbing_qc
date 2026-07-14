from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

AXES = ["음질", "감정 표현", "싱크 정확도", "자연스러움", "언어 적합성"]


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


class QCResult(BaseModel):
    verdict: Verdict
    findings: List[QCFinding]
    pairs: List[AlignedPair]


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


# Legacy models - kept for backward compatibility with existing endpoints
class ScriptSegment(BaseModel):
    id: str = Field(..., description="Segment ID")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker identifier")
    original_text: str = Field(..., description="Original Korean script line")
    translated_text: str = Field(..., description="English translated script line")

class QCRequest(BaseModel):
    video_url: Optional[str] = Field(None, description="Optional path or URL to video file")
    audio_path: Optional[str] = Field(None, description="Optional path to extracted audio file on server")
    segments: List[ScriptSegment] = Field(..., description="List of script segments to analyze")
    use_mock: bool = Field(True, description="Whether to use mock data for testing")

class QCStats(BaseModel):
    total_findings: int
    high_severity: int
    medium_severity: int
    low_severity: int
    localization_issues: int
    voice_issues: int

class QCResponse(BaseModel):
    overall_score: int = Field(..., description="Quality Score from 0 to 100")
    stats: QCStats = Field(..., description="Summary statistics of findings")
    findings: List[QCFinding] = Field(..., description="List of detected QC issues")
