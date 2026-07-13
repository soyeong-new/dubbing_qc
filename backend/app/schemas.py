from pydantic import BaseModel, Field
from typing import List, Optional

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

class QCFinding(BaseModel):
    id: str = Field(..., description="Finding ID")
    segment_id: str = Field(..., description="Associated segment ID")
    category: str = Field(..., description="Category: 'localization' or 'voice'")
    severity: str = Field(..., description="Severity: 'high', 'medium', or 'low'")
    issue_type: str = Field(..., description="Type of issue (e.g., 'Sync Error', 'Cultural Nuance', 'Voice Tone', 'Translation Mistake')")
    start_time: float = Field(..., description="Start time of the issue")
    end_time: float = Field(..., description="End time of the issue")
    speaker: str = Field(..., description="Speaker identifier")
    description: str = Field(..., description="Detailed description of the issue")
    original_text: str = Field(..., description="Korean text")
    current_translation: str = Field(..., description="Current English translation")
    recommendation: str = Field(..., description="AI suggested correction")
    confidence: float = Field(..., description="AI confidence score (0.0 to 1.0)")

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
