import asyncio
import re
from typing import List, Optional
from app.schemas import SegmentText

_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(content: str) -> List[SegmentText]:
    segments = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        time_idx = next((i for i, ln in enumerate(lines) if _TIME_RE.search(ln)), None)
        if time_idx is None:
            continue
        m = _TIME_RE.search(lines[time_idx])
        text = " ".join(lines[time_idx + 1:]).strip()
        if not text:
            continue
        segments.append(SegmentText(
            start=_to_seconds(*m.groups()[0:4]),
            end=_to_seconds(*m.groups()[4:8]),
            text=text,
        ))
    return segments


async def load_text_source(lang: str, srt_path: Optional[str],
                           audio_path: Optional[str]) -> List[SegmentText]:
    if srt_path:
        with open(srt_path, encoding="utf-8-sig") as f:
            return parse_srt(f.read())
    if audio_path and lang == "ko":
        from app.core.local_stt import transcribe_korean
        # transcribe_korean은 transformers 파이프라인을 동기로 호출한다 — asyncio
        # 이벤트 루프를 막지 않도록 스레드로 넘긴다.
        return await asyncio.to_thread(transcribe_korean, audio_path)
    raise ValueError(f"{lang}: SRT 또는 지원되는 오디오 STT 경로가 필요합니다.")
