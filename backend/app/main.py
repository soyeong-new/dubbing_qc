import os
import sys
import json

# Load environment variables from .env file if it exists at startup
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().strip("\"'")
                    os.environ[key] = val

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import QCRequest, QCResponse, ScriptSegment
from app.core.pipeline import QCPipeline
from typing import List
import shutil
import tempfile
import subprocess
import struct

app = FastAPI(title="AI Dubbing QC API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = QCPipeline()

@app.get("/")
def read_root():
    return {"message": "AI Dubbing QC Backend API is running."}

from pydantic import BaseModel

class TranscribeRequest(BaseModel):
    audio_path: str

class TranslateRequest(BaseModel):
    segments: List[ScriptSegment]

@app.post("/api/qc/upload-media")
async def upload_media(file: UploadFile = File(...), role: str = "dubbed"):
    """
    Saves an uploaded media file (video or audio), extracts standard audio,
    and returns 100Hz audio waveform peaks for the frontend visualization.
    role can be "original" (for STT/source) or "dubbed" (for Voice QC/target).
    """
    temp_dir = tempfile.gettempdir()
    # Use role and clean filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    media_path = os.path.join(temp_dir, f"{role}_{safe_filename}")
    
    with open(media_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    audio_filename = f"{role}_{os.path.splitext(safe_filename)[0]}.wav"
    audio_path = os.path.join(temp_dir, audio_filename)
    raw_audio_path = os.path.join(temp_dir, f"{role}_{os.path.splitext(safe_filename)[0]}_100hz.raw")
    
    try:
        # Extract audio (16kHz mono WAV) for Gemini analysis/transcription
        subprocess.run([
            "ffmpeg", "-i", media_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Extract 100Hz PCM raw audio for waveform peaks calculation
        subprocess.run([
            "ffmpeg", "-i", media_path,
            "-f", "s16le", "-ac", "1", "-ar", "100",
            "-y", raw_audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        peaks = []
        if os.path.exists(raw_audio_path):
            with open(raw_audio_path, "rb") as f:
                raw_data = f.read()
            
            num_samples = len(raw_data) // 2
            if num_samples > 0:
                samples = struct.unpack(f"{num_samples}h", raw_data)
                
                target_points = 600
                bin_size = max(1, num_samples // target_points)
                
                for i in range(0, num_samples, bin_size):
                    chunk = samples[i:i+bin_size]
                    if chunk:
                        max_val = max(abs(s) for s in chunk)
                        peaks.append(round(max_val / 32768.0, 3))
                        
            try:
                os.remove(raw_audio_path)
            except Exception:
                pass
                
        return {
            "success": True,
            "role": role,
            "filename": file.filename,
            "audio_path": audio_path,
            "media_path": media_path,
            "waveform": peaks
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/qc/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Backward-compatible endpoint for uploading dubbed video.
    """
    return await upload_media(file, role="dubbed")

@app.post("/api/qc/transcribe", response_model=List[ScriptSegment])
async def transcribe_media(request: TranscribeRequest):
    """
    Listens to the uploaded Korean audio file and generates Korean subtitles
    with start/end timestamps using Gemini 3.5 Flash STT.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback to sample transcription if API key is not set
        return [
            ScriptSegment(
                id="seg_1",
                start_time=1.0,
                end_time=4.5,
                speaker="형사 (Detective)",
                original_text="임마, 너 어제 김사장 만나서 눈치 보며 기어 다녔다며?",
                translated_text=""
            ),
            ScriptSegment(
                id="seg_2",
                start_time=5.2,
                end_time=7.8,
                speaker="용의자 (Suspect)",
                original_text="제가 왜 기어 다닙니까? 억울합니다 진짜. 밥도 못 먹고 조사받고 있어요.",
                translated_text=""
            ),
            ScriptSegment(
                id="seg_3",
                start_time=8.5,
                end_time=12.0,
                speaker="형사 (Detective)",
                original_text="참나, 어이가 없네. 니가 어제 클럽에서 돈 가방 들고 튀는 거 cctv에 다 찍혔어.",
                translated_text=""
            ),
            ScriptSegment(
                id="seg_4",
                start_time=13.0,
                end_time=14.8,
                speaker="용의자 (Suspect)",
                original_text="그거 제 가방 아닙니다. 진짜 아니에요. 살려주세요.",
                translated_text=""
            )
        ]
        
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    try:
        # Compress audio file to highly-efficient low-bitrate MP3 to avoid payload limits and speed up upload
        print(f"[STT 전사] 오디오 압축 시작 (WAV -> MP3)...")
        compressed_mp3 = os.path.join(tempfile.gettempdir(), f"compressed_{os.path.basename(request.audio_path)}.mp3")
        subprocess.run([
            "ffmpeg", "-i", request.audio_path,
            "-acodec", "libmp3lame", "-b:a", "24k", "-ar", "16000", "-ac", "1",
            "-y", compressed_mp3
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open(compressed_mp3, "rb") as f:
            audio_data = f.read()
            
        try:
            os.remove(compressed_mp3)
        except Exception:
            pass
            
        print(f"[STT 전사] 오디오 압축 완료. 전송 파일 크기: {len(audio_data) / 1024 / 1024:.2f} MB (원본 대비 약 15배 압축)")
        audio_part = {
            "mime_type": "audio/mp3",
            "data": audio_data
        }
        
        model = genai.GenerativeModel('gemini-3.5-flash')
        
        prompt = """
        제공된 한국어 오디오 파일을 듣고, 화자별 대사 내용과 시작/종료 시간(초 단위)을 추출하여 한국어 전사를 작성해 주십시오.
        반드시 아래 JSON 배열 포맷으로 결과를 반환하십시오. 각 세그먼트는 발화 내용 단위로 분할하되, 하나의 세그먼트가 너무 길지 않게(보통 1~4초 내외) 세분화해 주십시오.
        
        주의: translated_text 필드는 빈 문자열("")로 반환해 주십시오. id는 "seg_1", "seg_2"와 같이 순차적으로 부여해 주십시오.
        
        반환할 JSON 객체 스키마:
        [
          {
            "id": "seg_1",
            "start_time": 1.2,
            "end_time": 4.5,
            "speaker": "인물 1",
            "original_text": "한국어 전사 내용",
            "translated_text": ""
          }
        ]
        """
        
        print("[STT 전사] Gemini 3.5 Flash 멀티모달 STT 분석 호출 중 (음원을 직접 분석합니다)...")
        response = model.generate_content(
            [audio_part, prompt],
            generation_config={"response_mime_type": "application/json"}
        )
        print("[STT 전사] Gemini 응답 수신 완료. JSON 자막 파싱 시작...")
        
        segments_data = json.loads(response.text)
        
        res_segments = []
        for i, item in enumerate(segments_data):
            res_segments.append(ScriptSegment(
                id=item.get("id", f"seg_{i+1}"),
                start_time=float(item.get("start_time", 0.0)),
                end_time=float(item.get("end_time", 0.0)),
                speaker=item.get("speaker", "인물"),
                original_text=item.get("original_text", ""),
                translated_text=""
            ))
        print(f"[STT 전사] 전사 완료! 총 {len(res_segments)}개의 한글 자막 세그먼트 생성 성공.\n")
        return res_segments
    except Exception as e:
        print(f"[STT 전사] 오류 발생: {e}")
        return [
            ScriptSegment(
                id="seg_1",
                start_time=1.0,
                end_time=4.5,
                speaker="형사 (Detective)",
                original_text="[STT 실패 대체 대사] 임마, 너 어제 김사장 만나서 눈치 보며 기어 다녔다며?",
                translated_text=""
            )
        ]

@app.post("/api/qc/translate", response_model=List[ScriptSegment])
async def translate_script(request: TranslateRequest):
    """
    Translates Korean script segments to English using Gemini 3.5 Flash,
    preserving speakers and timestamps.
    """
    if not request.segments:
        print("[AI 번역] 오류: 번역할 자막 세그먼트가 없습니다.")
        return []
        
    print(f"\n[AI 번역] >>> 한국어 자막 -> 영어 더빙 대본 번역 프로세스 시작...")
    print(f"[AI 번역] 번역 대상 세그먼트 개수: {len(request.segments)}개")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        translated = []
        for seg in request.segments:
            text_kr = seg.original_text
            text_en = f"[AI Translation] {text_kr}"
            if "눈치" in text_kr:
                text_en = "Hey kid, didn't you meet President Kim yesterday and crawl while looking at eyes?"
            elif "억울합니다" in text_kr:
                text_en = "Why do I crawl? I am truly unfair. I can't even eat rice and am being investigated."
            elif "어이가 없네" in text_kr:
                text_en = "Wow, I have no kidney. You were filmed on CCTV running with a bag of money."
            
            translated.append(ScriptSegment(
                id=seg.id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker=seg.speaker,
                original_text=seg.original_text,
                translated_text=text_en
            ))
        return translated

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-3.5-flash')
        
        prompt = """
        당신은 한국 영화의 영어 더빙 자막을 번역하는 전문 번역 에이전트입니다.
        제공되는 한국어 영화 자막 세그먼트 목록을 보고, 영어 더빙 성우들이 연기하기에 가장 적절하고 자연스러운 구어체 영어 대사로 번역해 주십시오.

        번역 지침:
        1. 한국의 독특한 문화적 맥락(예: 호칭 '형', '누나', '부장님' 등 또는 일상적 안부 '밥 먹었니' 등)이 직역되어 어색해지지 않도록 자연스럽게 영어 표현으로 의역해 주십시오.
        2. 자막 글자 수 및 발화 제한 시간을 감안하여 번역어가 너무 장황해지지 않게 핵심 의미를 간결하게 표현해 주십시오.
        3. 각 세그먼트의 id, start_time, end_time, speaker는 원본과 정확히 동일하게 유지하십시오.
        4. 반드시 아래 JSON 배열 스키마에 맞춰 결과를 반환해 주십시오.

        반환할 JSON 객체 스키마:
        [
          {
            "id": "seg_id",
            "start_time": 1.2,
            "end_time": 4.5,
            "speaker": "인물명",
            "original_text": "한국어 원문",
            "translated_text": "번역된 영어 대사"
          }
        ]

        번역할 자막 세그먼트 목록:
        """
        
        payload = []
        for seg in request.segments:
            payload.append({
                "id": seg.id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker": seg.speaker,
                "original_text": seg.original_text,
                "translated_text": ""
            })
            
        prompt += json.dumps(payload, ensure_ascii=False, indent=2)
        
        print("[AI 번역] Gemini 3.5 Flash 번역 호출 중...")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        print("[AI 번역] Gemini 번역 응답 수신 완료. 결과 파싱 중...")
        
        translated_data = json.loads(response.text)
        
        res_segments = []
        for item in translated_data:
            res_segments.append(ScriptSegment(
                id=item.get("id"),
                start_time=float(item.get("start_time", 0.0)),
                end_time=float(item.get("end_time", 0.0)),
                speaker=item.get("speaker", ""),
                original_text=item.get("original_text", ""),
                translated_text=item.get("translated_text", "")
            ))
        return res_segments
    except Exception as e:
        print(f"Translation Error: {e}")
        return [
            ScriptSegment(
                id=seg.id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker=seg.speaker,
                original_text=seg.original_text,
                translated_text=f"Dubbed: {seg.original_text}"
            ) for seg in request.segments
        ]

@app.post("/api/qc/process", response_model=QCResponse)
async def process_qc(request: QCRequest):
    return await pipeline.run(request)

@app.get("/api/qc/mock-data", response_model=List[ScriptSegment])
def get_mock_data():
    """
    Returns a set of mock script segments representing typical Korean movie dialogues
    with common AI translation/dubbing errors for demonstration.
    """
    return [
        ScriptSegment(
            id="seg_1",
            start_time=0.5,
            end_time=3.2,
            speaker="민우 (Min-woo)",
            original_text="야, 너 어제 형한테 눈치 보지 말고 말하라고 했지?",
            translated_text="Hey, didn't you tell brother yesterday to speak without looking at eyes?"
        ),
        ScriptSegment(
            id="seg_2",
            start_time=4.0,
            end_time=5.5,
            speaker="수현 (Su-hyun)",
            original_text="내가 언제 그랬어? 미안하게 진짜.",
            translated_text="When did I say that? I am truly sorry."
        ),
        ScriptSegment(
            id="seg_3",
            start_time=6.0,
            end_time=7.5,
            speaker="민우 (Min-woo)",
            original_text="참나, 어이가 없네. 밥은 먹었냐?",
            translated_text="Wow, I have no kidney. Did you eat rice?"
        ),
        ScriptSegment(
            id="seg_4",
            start_time=8.0,
            end_time=9.8,
            speaker="수현 (Su-hyun)",
            original_text="갑자기 무슨 밥 타령이야? 비켜, 나 바빠.",
            translated_text="Why are you talking about rice all of a sudden? Get out of my way, I am very busy right now because I have a lot of things to finish before the sun goes down."
        )
    ]
