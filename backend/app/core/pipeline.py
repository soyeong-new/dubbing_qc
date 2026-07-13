from typing import List
from app.schemas import QCRequest, QCResponse, QCFinding, QCStats
from app.core.context import ContextEngine
from app.core.localization import LocalizationEngine
from app.core.voice_qc import VoiceQCEngine

class QCPipeline:
    def __init__(self):
        self.context_engine = ContextEngine()
        self.localization_engine = LocalizationEngine()
        self.voice_engine = VoiceQCEngine()

    async def run(self, request: QCRequest) -> QCResponse:
        # 1. Context extraction
        context_map = await self.context_engine.analyze_scenes(request.segments)
        
        # 2. Localization QC
        loc_findings = await self.localization_engine.analyze(
            request.segments, 
            context_map, 
            audio_path=request.audio_path,
            use_mock=request.use_mock
        )
        
        # 3. Voice QC
        voice_findings = await self.voice_engine.analyze(
            request.segments, 
            context_map, 
            audio_path=request.audio_path,
            use_mock=request.use_mock
        )
        
        # Combine all findings
        all_findings = loc_findings + voice_findings
        
        # 4. Compute statistics
        high_cnt = sum(1 for f in all_findings if f.severity == "high")
        med_cnt = sum(1 for f in all_findings if f.severity == "medium")
        low_cnt = sum(1 for f in all_findings if f.severity == "low")
        loc_cnt = sum(1 for f in all_findings if f.category == "localization")
        voice_cnt = sum(1 for f in all_findings if f.category == "voice")
        
        # Quality score formula (starts at 100, drops by severity weight)
        # High = -15 pts, Medium = -8 pts, Low = -3 pts
        deduction = (high_cnt * 15) + (med_cnt * 8) + (low_cnt * 3)
        overall_score = max(100 - deduction, 0)
        
        stats = QCStats(
            total_findings=len(all_findings),
            high_severity=high_cnt,
            medium_severity=med_cnt,
            low_severity=low_cnt,
            localization_issues=loc_cnt,
            voice_issues=voice_cnt
        )
        
        return QCResponse(
            overall_score=overall_score,
            stats=stats,
            findings=all_findings
        )
