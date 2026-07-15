import os
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas import SegmentText, AlignedPair, QCFinding


class ProviderNotConfiguredError(RuntimeError):
    pass


class Persona(BaseModel):
    key: str
    name: str
    instruction: str
    uses_audio: bool = False
    axes: List[str] = Field(default_factory=list)


class ModelProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: str, lang: str) -> List[SegmentText]:
        ...

    @abstractmethod
    async def judge(self, pairs: List[AlignedPair], persona: Persona,
                    knowledge: str, audio_clip_path: Optional[str] = None,
                    original_audio_clip_path: Optional[str] = None) -> List[QCFinding]:
        ...

    # score(pairs) -> list[float] : 2단계 xCOMET 스크리너용 예약. 이번 단계에서 구현하지 않음.


def get_provider() -> ModelProvider:
    name = os.getenv("QC_PROVIDER", "gemini")
    if name == "mock":
        # 검수 결과에 목 데이터가 섞이면 안 된다 — 테스트 프로세스에서만 허용
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise ProviderNotConfiguredError("mock 프로바이더는 자동화 테스트 전용입니다.")
        from app.providers.mock import MockProvider
        return MockProvider()
    if name == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise ProviderNotConfiguredError(
                "GEMINI_API_KEY가 설정되지 않았습니다. 검수를 시작할 수 없습니다. "
                ".env 파일 또는 환경변수에 키를 설정하세요."
            )
        from app.providers.gemini import GeminiProvider
        return GeminiProvider()
    raise ProviderNotConfiguredError(f"알 수 없는 프로바이더: {name}")
