import os
import json
from typing import List, Dict
from app.schemas import ScriptSegment, QCFinding
import google.generativeai as genai

class LocalizationEngine:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    async def analyze(self, segments: List[ScriptSegment], context_map: Dict[str, str], audio_path: str = None, use_mock: bool = True) -> List[QCFinding]:
        """
        Analyzes translation naturalness, cultural risks, tone, and humor.
        Returns a list of QCFindings.
        """
        if not use_mock and self.api_key:
            return await self._analyze_with_gemini(segments, context_map)
        else:
            return self._generate_mock_findings(segments, context_map)

    async def _analyze_with_gemini(self, segments: List[ScriptSegment], context_map: Dict[str, str]) -> List[QCFinding]:
        findings = []
        try:
            model = genai.GenerativeModel('gemini-3.5-flash')
            
            prompt = """
            당신은 한국 영화의 영어 더빙 자막을 검수하는 전문 AI QC 에이전트입니다.
            제공되는 자막 세그먼트들을 분석하여 영어 더빙 대사로서 부적절하거나 수정이 필요한 항목들을 찾아내십시오.

            다음 기준에 따라 검수하십시오:
            1. 번역 오류: 직역으로 인한 오역이나 텍스트 맥락에 어긋나는 기계 번역 에러가 있는지 (예: '어이가 없네' -> 'no kidney')
            2. 문화적 정서 차이: 한국 특유의 호칭(형, 누나, 오빠, 부장님 등)이나 관용구(눈치, 밥 타령 등)가 영어로 너무 곧이곧대로 번역되어 어색한지
            3. 톤/어조 불일치: 영상의 맥락이나 캐릭터 성격에 비해 문체가 어색한지
            4. 부자연스러운 표현: 문장 자체가 원어민이 쓰기에 너무 번역투이거나 어색한지
            5. 번역 누락: 영문 번역 대본이 누락되어 비어 있는지

            각 문제점에 대해 반드시 다음 형식을 갖춘 JSON 배열을 반환하십시오.
            **주의**: 대사의 문제점이 왜 일어났고 왜 수정해야 하는지에 대한 분석 설명(`description`)은 반드시 **한국어**로 작성되어야 합니다. 수정 추천 대사(`recommendation`)는 최종적으로 더빙에 사용할 **영어**여야 합니다.

            반환할 JSON 객체 스키마:
            - segment_id: 해당 세그먼트 ID
            - severity: 위험 강도 ("high" | "medium" | "low")
            - issue_type: 오류 종류 ("번역 오류" | "문화적 정서 차이" | "톤/어조 불일치" | "부자연스러운 표현" | "번역 누락")
            - description: 왜 대사를 영어 더빙 관점에서 수정해야 하는지 이유와 문제점을 친절하게 설명 (반드시 한국어로 작성!)
            - recommendation: 추천하는 최종 영어 더빙 대사 (반드시 수정안으로 교체 가능한 영어 문장으로 작성)
            - confidence: 판단 신뢰도 수치 (0.0에서 1.0 사이의 실수)

            수정이 필요 없는 정상적인 라인은 결과에 포함하지 마십시오.

            분석할 자막 데이터 목록:
            """
            
            payload = []
            for seg in segments:
                payload.append({
                    "id": seg.id,
                    "speaker": seg.speaker,
                    "korean": seg.original_text,
                    "english": seg.translated_text,
                    "scene_context": context_map.get(seg.id, "")
                })
            
            prompt += json.dumps(payload, ensure_ascii=False, indent=2)
            
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            results = json.loads(response.text)
            for res in results:
                seg = next((s for s in segments if s.id == res["segment_id"]), None)
                if seg:
                    findings.append(QCFinding(
                        id=f"loc_{seg.id}_{res.get('issue_type', 'error').lower().replace(' ', '_')}",
                        segment_id=seg.id,
                        category="localization",
                        severity=res.get("severity", "medium"),
                        issue_type=res.get("issue_type", "번역 오류"),
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        speaker=seg.speaker,
                        description=res.get("description", ""),
                        original_text=seg.original_text,
                        current_translation=seg.translated_text,
                        recommendation=res.get("recommendation", ""),
                        confidence=res.get("confidence", 0.85)
                    ))
        except Exception as e:
            print(f"Gemini API Error: {e}. Falling back to rule-based engine.")
            return self._generate_mock_findings(segments, context_map)
            
        return findings

    def _generate_mock_findings(self, segments: List[ScriptSegment], context_map: Dict[str, str]) -> List[QCFinding]:
        findings = []
        
        for seg in segments:
            text_kr = seg.original_text or ""
            text_en = (seg.translated_text or "").strip()
            
            # Check 0: Missing Translation
            if not text_en:
                findings.append(QCFinding(
                    id=f"loc_{seg.id}_missing",
                    segment_id=seg.id,
                    category="localization",
                    severity="high",
                    issue_type="번역 누락",
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=seg.speaker,
                    description="해당 대사 영역의 영문 번역 대본이 누락되었습니다. 더빙 오디오 매핑 및 검수를 위해 대사 번역을 먼저 채워야 합니다.",
                    original_text=seg.original_text,
                    current_translation="",
                    recommendation="AI recommended dubbing line here.",
                    confidence=1.0
                ))
                continue
            
            # Check 1: Honorifics
            if any(h in text_kr for h in ["형", "누나", "오빠", "언니", "부장님", "선배"]) and any(e in text_en.lower() for e in ["brother", "sister", "director", "senior"]):
                findings.append(QCFinding(
                    id=f"loc_{seg.id}_honorific",
                    segment_id=seg.id,
                    category="localization",
                    severity="medium",
                    issue_type="문화적 정서 차이",
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=seg.speaker,
                    description="한국어 친족 호칭(형/누나/오빠)이나 직책이 영어 'brother', 'sister', 'director' 등으로 직역되어 대화 맥락상 어색합니다. 친근한 대화 톤에 맞춰 일반적인 이름이나 자연스러운 호칭으로 의역하는 것을 추천합니다.",
                    original_text=seg.original_text,
                    current_translation=seg.translated_text,
                    recommendation=seg.translated_text.replace("Brother", "Hey").replace("brother", "man").replace("Senior", "").replace("Director", "Mr. Han"),
                    confidence=0.92
                ))

            # Check 2: Literal translation of "밥 먹었어?"
            elif "밥" in text_kr and "rice" in text_en.lower():
                findings.append(QCFinding(
                    id=f"loc_{seg.id}_literal_rice",
                    segment_id=seg.id,
                    category="localization",
                    severity="low",
                    issue_type="문화적 정서 차이",
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=seg.speaker,
                    description="'밥 먹었어?'는 한국어 안부 인사말(How are you? / Have you eaten?)입니다. 이를 'Did you eat rice?'로 번역하여 의미상 어색함이 발생하므로 자연스러운 대화형 안부 문구로 수정해 주세요.",
                    original_text=seg.original_text,
                    current_translation=seg.translated_text,
                    recommendation="Did you get a bite to eat?" if "eat" in text_en.lower() else "How are you doing?",
                    confidence=0.95
                ))
            
            # Check 3: Literal translation of "눈치"
            elif "눈치" in text_kr and ("eye" in text_en.lower() or "look" in text_en.lower()):
                findings.append(QCFinding(
                    id=f"loc_{seg.id}_nunchi",
                    segment_id=seg.id,
                    category="localization",
                    severity="high",
                    issue_type="번역 오류",
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=seg.speaker,
                    description="'눈치 보다'라는 고유 개념이 눈(eyes/look)으로 직접 번역되었습니다. 상황과 긴장감 수준에 따라 'walking on eggshells'(조심조심 살피다) 또는 'read the room'(맥락 파악) 등으로 변경해야 합니다.",
                    original_text=seg.original_text,
                    current_translation=seg.translated_text,
                    recommendation="Stop walking on eggshells." if "보지" in text_kr else "You should read the room.",
                    confidence=0.88
                ))

            # Check 4: Literal translation of "어이가 없네"
            elif "어이가 없네" in text_kr and "kidney" in text_en.lower():
                findings.append(QCFinding(
                    id=f"loc_{seg.id}_kidney",
                    segment_id=seg.id,
                    category="localization",
                    severity="high",
                    issue_type="번역 오류",
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=seg.speaker,
                    description="관용구 '어이가 없네'(황당하다)의 '어이'가 인체 장기인 신장(kidney)으로 치명적인 기계 번역 오역이 일어났습니다. 황당함을 표현하는 영어단어(예: ridiculous, speechless)로 즉시 교정하십시오.",
                    original_text=seg.original_text,
                    current_translation=seg.translated_text,
                    recommendation="This is ridiculous.",
                    confidence=0.99
                ))

            # Check 5: General pacing check
            elif len(text_en.split()) > 10 and seg.end_time - seg.start_time < 2.0:
                findings.append(QCFinding(
                    id=f"loc_{seg.id}_pacing",
                    segment_id=seg.id,
                    category="localization",
                    severity="medium",
                    issue_type="부자연스러운 표현",
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker=seg.speaker,
                    description="자막 제한 시간(2초 미만) 대비 번역된 문장의 길이가 지나치게 길어 성우가 대사를 읽을 때 싱크 매칭이 어렵습니다. 간결한 표현으로 대본을 다듬어야 합니다.",
                    original_text=seg.original_text,
                    current_translation=seg.translated_text,
                    recommendation=" ".join(text_en.split()[:5]) + "...",
                    confidence=0.80
                ))
                
        return findings
