import asyncio
import os
import json
import subprocess
import tempfile
from typing import List, Optional
from app.providers.base import ModelProvider, Persona
from app.schemas import AlignedPair, QCFinding, AXES

MODEL_NAME = "gemini-3.5-flash"

JUDGE_PROMPT_TEMPLATE = """
당신은 한국 영화의 영어 더빙을 검수하는 "{persona_name}"입니다.

{persona_instruction}

## 검수 지식베이스 (참고 규칙)
{knowledge}

## 지시
아래 세그먼트 쌍(한국어 원문 ↔ 영어 더빙)을 검토하여 문제가 있는 항목만 JSON 배열로 반환하십시오.
- description은 반드시 **한국어**로: 무엇이 왜 문제인지 설명
- recommendation은 반드시 **영어**로: 교체 가능한 최종 더빙 대사
- axis는 다음 중 하나: {axes}
- severity는 "high" | "medium" | "low"
- finding_type은 "quality"(일반 품질 지적) 또는 "sensitive"(민감어·욕설 지적) 중 하나.
  생략 시 "quality"로 간주됩니다.
- 문제 없는 세그먼트는 결과에 포함하지 마십시오.

반환 스키마:
[{{"segment_id": "...", "severity": "...", "issue_type": "...",
  "description": "...", "recommendation": "...", "confidence": 0.9, "axis": "...",
  "finding_type": "quality"}}]

## 분석할 세그먼트 쌍
{payload}
"""


def build_judge_prompt(pairs: List[AlignedPair], persona: Persona, knowledge: str) -> str:
    payload = []
    for p in pairs:
        payload.append({
            "segment_id": p.id,
            "korean": p.korean.text if p.korean else "",
            "english_dub": p.dubbed.text if p.dubbed else "",
            "speaker": p.korean.speaker if p.korean else (p.dubbed.speaker if p.dubbed else "?"),
            "start": p.korean.start if p.korean else (p.dubbed.start if p.dubbed else 0),
            "end": p.korean.end if p.korean else (p.dubbed.end if p.dubbed else 0),
        })
    return JUDGE_PROMPT_TEMPLATE.format(
        persona_name=persona.name,
        persona_instruction=persona.instruction,
        knowledge=knowledge or "(등록된 규칙 없음)",
        axes=" | ".join(persona.axes or AXES),
        payload=json.dumps(payload, ensure_ascii=False, indent=1),
    )


def parse_judge_response(text: str, pairs: List[AlignedPair], persona: Persona) -> List[QCFinding]:
    by_id = {p.id: p for p in pairs}
    default_axis = persona.axes[0] if persona.axes else "언어 적합성"
    findings = []
    for i, item in enumerate(json.loads(text)):
        pair = by_id.get(item.get("segment_id"))
        if pair is None:
            continue
        axis = item.get("axis", default_axis)
        if axis not in AXES:
            axis = default_axis
        # 영어 SRT가 타임코드 기준(주체)이다 — 한국어 STT 경계는 환각/부정확한
        # 타이밍을 포함할 수 있어 화면 표시·영상 이동에는 쓰지 않는다(실측 확인).
        anchor = pair.dubbed or pair.korean
        finding_type = item.get("finding_type", "quality")
        if finding_type not in ("quality", "sensitive"):
            finding_type = "quality"
        findings.append(QCFinding(
            id=f"{persona.key}_{pair.id}_{i}",
            segment_id=pair.id,
            category="localization",
            severity=item.get("severity", "medium"),
            issue_type=item.get("issue_type", "번역 오류"),
            start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
            description=item.get("description", ""),
            original_text=pair.korean.text if pair.korean else "",
            current_translation=pair.dubbed.text if pair.dubbed else "",
            recommendation=item.get("recommendation", ""),
            confidence=float(item.get("confidence", 0.8)),
            axis=axis,
            source=f"persona:{persona.key}",
            finding_type=finding_type,
        ))
    return findings


def _compress_to_mp3(audio_path: str) -> bytes:
    out = os.path.join(tempfile.gettempdir(), f"qc_compress_{os.getpid()}.mp3")
    subprocess.run(
        ["ffmpeg", "-i", audio_path, "-acodec", "libmp3lame",
         "-b:a", "24k", "-ar", "16000", "-ac", "1", "-y", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    with open(out, "rb") as f:
        data = f.read()
    os.remove(out)
    return data


class GeminiProvider(ModelProvider):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._genai = genai

    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        model = self._genai.GenerativeModel(MODEL_NAME)
        prompt = build_judge_prompt(pairs, persona, knowledge)
        parts = [prompt]
        if audio_clip_path and persona.uses_audio and os.path.exists(audio_clip_path):
            dub_audio = await asyncio.to_thread(_compress_to_mp3, audio_clip_path)
            parts.insert(0, {"mime_type": "audio/mp3", "data": dub_audio})
            parts.insert(0, "[다음 오디오는 영어 더빙입니다]")
        if original_audio_clip_path and persona.uses_audio and os.path.exists(original_audio_clip_path):
            orig_audio = await asyncio.to_thread(_compress_to_mp3, original_audio_clip_path)
            parts.insert(0, {"mime_type": "audio/mp3", "data": orig_audio})
            parts.insert(0, "[다음 오디오는 한국어 원본입니다]")
        # 동기 SDK 호출을 스레드로 넘겨 이벤트 루프가 다른 요청(진행률 폴링 등)을
        # 계속 처리할 수 있게 한다.
        response = await asyncio.to_thread(
            model.generate_content,
            parts, generation_config={"response_mime_type": "application/json"},
        )
        return parse_judge_response(response.text, pairs, persona)
