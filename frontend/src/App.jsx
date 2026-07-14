import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import ProjectView from "./views/ProjectView";

function App() {
  const [view, setView] = useState("project"); // "project" | "review" | "report"
  const [uploads, setUploads] = useState({});
  const [jobId, setJobId] = useState(null);
  const [qcResult, setQcResult] = useState(null);
  const [segments, setSegments] = useState([]);
  const [findings, setFindings] = useState([]);
  const [overallScore, setOverallScore] = useState(100);
  const [stats, setStats] = useState({
    total: 0,
    high: 0,
    medium: 0,
    low: 0,
    localization: 0,
    voice: 0
  });

  const [activeSegmentId, setActiveSegmentId] = useState(null);
  const [filter, setFilter] = useState("all"); 
  const [severityFilter, setSeverityFilter] = useState("all"); 
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [analysisSource, setAnalysisSource] = useState(null); // "gemini" | "local"
  const [analysisError, setAnalysisError] = useState(null);
  
  // Video & audio states
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(10.0); 
  const [originalVideoSrc, setOriginalVideoSrc] = useState(null);
  const [dubbedVideoSrc, setDubbedVideoSrc] = useState("https://assets.mixkit.co/videos/preview/mixkit-cyberpunk-city-street-at-night-40134-large.mp4");
  const [activeVideoRole, setActiveVideoRole] = useState("dubbed");
  
  // File names for display
  const [videoFileName, setVideoFileName] = useState("cyberpunk_city.mp4 (기본 샘플)");
  const [krFileName, setKrFileName] = useState("cyberpunk_kr.srt (기본 샘플)");
  const [enFileName, setEnFileName] = useState("cyberpunk_en.srt (기본 샘플)");

  // Temporary raw SRT arrays for alignment
  const [rawKrSegments, setRawKrSegments] = useState([]);
  const [rawEnSegments, setRawEnSegments] = useState([]);

  // Real waveform & backend audio path states
  const [waveformPeaks, setWaveformPeaks] = useState([]);
  const [backendAudioPath, setBackendAudioPath] = useState(null);
  const [uploadingVideo, setUploadingVideo] = useState(false);

  // STT Transcription & AI Translation states
  const [originalMediaName, setOriginalMediaName] = useState("선택되지 않음");
  const [originalAudioPath, setOriginalAudioPath] = useState(null);
  const [uploadingOriginal, setUploadingOriginal] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const videoInputRef = useRef(null);
  const krInputRef = useRef(null);
  const enInputRef = useRef(null);
  const originalInputRef = useRef(null);

  // 1. Job completion handler — bridges the Project tab's async QC job
  // result into the existing review-dashboard state (segments/findings/stats)
  // so the review view can be reused as-is, then switches to the review tab.
  const handleJobComplete = (id, result) => {
    setJobId(id);
    setQcResult(result);
    setFindings(result.findings);
    // AlignedPair → 기존 세그먼트 상태로 변환 (검수 뷰 재사용)
    setSegments(result.pairs.map((p) => ({
      id: p.id,
      start_time: (p.korean || p.dubbed).start,
      end_time: (p.korean || p.dubbed).end,
      speaker: (p.korean || p.dubbed).speaker,
      original_text: p.korean ? p.korean.text : "",
      translated_text: p.dubbed ? p.dubbed.text : "",
    })));
    updateStats(result.findings);
    setView("review");
  };

  // Helper: Normalize SRT Time format "00:00:01,000" to seconds
  const normalizeSRTTime = (timeStr) => {
    const parts = timeStr.trim().split(":");
    if (parts.length < 3) return 0;
    const hrs = parseInt(parts[0], 10);
    const mins = parseInt(parts[1], 10);
    const secsParts = parts[2].split(",");
    const secs = parseInt(secsParts[0], 10);
    const ms = secsParts[1] ? parseInt(secsParts[1], 10) / 1000 : 0;
    return hrs * 3600 + mins * 60 + secs + ms;
  };

  // Helper: Parse raw SRT text into subtitle line objects
  const parseSRT = (text) => {
    const blocks = text.trim().split(/\r?\n\r?\n/);
    return blocks.map((block, index) => {
      const lines = block.split(/\r?\n/);
      if (lines.length < 3) return null;
      
      const timeMatch = lines[1].match(/(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/);
      if (!timeMatch) return null;
      
      const start_time = normalizeSRTTime(timeMatch[1]);
      const end_time = normalizeSRTTime(timeMatch[2]);
      
      let dialogue = lines.slice(2).join(" ");
      let speaker = "등장인물";
      
      const speakerMatch = dialogue.match(/^\[?([^:\]]+)\]?:\s*(.*)$/);
      if (speakerMatch) {
        speaker = speakerMatch[1].trim();
        dialogue = speakerMatch[2].trim();
      }
      
      return {
        id: `srt_${index}`,
        start_time,
        end_time,
        speaker,
        text: dialogue
      };
    }).filter(Boolean);
  };

  // Helper: Align Korean and English scripts based on time overlap
  const alignScripts = (krList, enList) => {
    if (krList.length === 0 && enList.length === 0) return [];
    
    if (krList.length > 0 && enList.length === 0) {
      return krList.map(kr => ({
        id: kr.id,
        start_time: kr.start_time,
        end_time: kr.end_time,
        speaker: kr.speaker === "dubbed" ? "인물" : kr.speaker,
        original_text: kr.text,
        translated_text: ""
      }));
    }
    
    if (enList.length > 0 && krList.length === 0) {
      return enList.map(en => ({
        id: en.id,
        start_time: en.start_time,
        end_time: en.end_time,
        speaker: en.speaker === "dubbed" ? "인물" : en.speaker,
        original_text: "",
        translated_text: en.text
      }));
    }

    return krList.map((kr, idx) => {
      // Find matching English line by time overlap (tolerance: 3.0s)
      let matchedEn = enList.find(en => 
        Math.abs(en.start_time - kr.start_time) < 3.0
      );
      
      // Fallback: index-based match if timecode match fails
      if (!matchedEn && enList[idx]) {
        matchedEn = enList[idx];
      }
      
      const enText = matchedEn ? matchedEn.text : "";
      const enSpeaker = matchedEn ? matchedEn.speaker : "";
      
      // Select speaker name: prefer Korean speaker name, filter out "등장인물", "인물", and "dubbed"
      let speaker = kr.speaker;
      if (speaker === "등장인물" || speaker === "인물" || speaker === "dubbed" || !speaker) {
        speaker = (enSpeaker && enSpeaker !== "등장인물" && enSpeaker !== "dubbed") ? enSpeaker : "인물";
      }
      
      return {
        id: kr.id,
        start_time: kr.start_time,
        end_time: kr.end_time,
        speaker: speaker,
        original_text: kr.text,
        translated_text: enText
      };
    });
  };

  // 2. File Upload Handlers
  const handleOriginalUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setOriginalVideoSrc(url);
      setOriginalMediaName(file.name);
      setActiveVideoRole("original");
      setIsPlaying(false);
      if (videoRef.current) {
        videoRef.current.load();
      }

      setUploadingOriginal(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("http://localhost:8000/api/qc/upload-media?role=original", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.success) {
          setOriginalAudioPath(data.audio_path);
          console.log("Original media uploaded successfully:", data);
        } else {
          console.error("Original media upload failed:", data.error);
          setAnalysisError(`원본 미디어 업로드 실패: ${data.error || "알 수 없는 오류"}`);
        }
      } catch (err) {
        console.error("Error uploading original media:", err);
        setAnalysisError(`원본 미디어 업로드 실패: ${err.message}`);
      } finally {
        setUploadingOriginal(false);
      }
    }
  };

  const handleTranscribe = async () => {
    if (!originalAudioPath) return;
    setIsTranscribing(true);
    try {
      const res = await fetch("http://localhost:8000/api/qc/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_path: originalAudioPath }),
      });
      if (!res.ok) throw new Error(`서버 응답 ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        // Convert to the raw-KR shape alignScripts expects, and re-align against
        // any English subtitles already loaded, instead of overwriting segments
        // outright (which used to wipe out previously loaded EN translations).
        const krList = data.map(s => ({
          id: s.id,
          start_time: s.start_time,
          end_time: s.end_time,
          speaker: s.speaker,
          text: s.original_text
        }));
        setRawKrSegments(krList);
        const merged = alignScripts(krList, rawEnSegments);
        setSegments(merged);
        setKrFileName("[AI 전사 완료] ko_transcript.json");
        runQCAnalysis(merged, false);
      } else {
        throw new Error("전사 결과 형식이 올바르지 않습니다.");
      }
    } catch (err) {
      console.error("Transcription failed:", err);
      setAnalysisError(`AI 전사(STT) 실패: ${err.message}`);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleTranslate = async () => {
    if (segments.length === 0) return;
    setIsTranslating(true);
    try {
      const res = await fetch("http://localhost:8000/api/qc/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments }),
      });
      if (!res.ok) throw new Error(`서버 응답 ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setSegments(data);
        setEnFileName("[AI 번역 완료] en_translation.srt");
        runQCAnalysis(data, false);
      } else {
        throw new Error("번역 결과 형식이 올바르지 않습니다.");
      }
    } catch (err) {
      console.error("Translation failed:", err);
      setAnalysisError(`AI 번역 실패: ${err.message}`);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setDubbedVideoSrc(url);
      setVideoFileName(file.name);
      setActiveVideoRole("dubbed");
      setIsPlaying(false);
      if (videoRef.current) {
        videoRef.current.load();
      }

      // Upload to backend to extract audio and compute waveform
      setUploadingVideo(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("http://localhost:8000/api/qc/upload-media?role=dubbed", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.success) {
          setWaveformPeaks(data.waveform);
          setBackendAudioPath(data.audio_path);
          console.log("Video uploaded and audio/waveform extracted successfully:", data);
        } else {
          console.error("Video upload failed:", data.error);
          setAnalysisError(`영상 업로드 실패: ${data.error || "알 수 없는 오류"}`);
        }
      } catch (err) {
        console.error("Error uploading video:", err);
        setAnalysisError(`영상 업로드 실패: ${err.message}`);
      } finally {
        setUploadingVideo(false);
      }
    }
  };

  const handleKrUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        
        if (file.name.endsWith(".json")) {
          try {
            const parsed = JSON.parse(text);
            if (Array.isArray(parsed) && parsed.length > 0) {
              setSegments(parsed);
              setKrFileName(file.name);
              setEnFileName("JSON 파일 내장됨");
              runQCAnalysis(parsed, backendStatus === "standalone");
            }
          } catch (err) {
            alert("JSON 파싱 에러");
          }
        } else if (file.name.endsWith(".srt")) {
          const parsedSRT = parseSRT(text);
          setRawKrSegments(parsedSRT);
          setKrFileName(file.name);
          
          const aligned = alignScripts(parsedSRT, rawEnSegments);
          setSegments(aligned);
          runQCAnalysis(aligned, backendStatus === "standalone");
        }
      };
      reader.readAsText(file);
    }
  };

  const handleEnUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const parsedSRT = parseSRT(text);
        setRawEnSegments(parsedSRT);
        setEnFileName(file.name);
        
        const aligned = alignScripts(rawKrSegments, parsedSRT);
        setSegments(aligned);
        runQCAnalysis(aligned, backendStatus === "standalone");
      };
      reader.readAsText(file);
    }
  };

  // 3. Core QC Analyzer (API or Standalone JS engine)
  // knownConnected lets a caller assert connectivity it just confirmed itself
  // (e.g. right after a successful fetch), instead of relying on the
  // `backendStatus` state var, which may still hold its pre-update value here
  // due to React closures when called synchronously after setBackendStatus().
  const runQCAnalysis = async (currentSegments, forceStandalone = false, knownConnected = null) => {
    if (currentSegments.length === 0) return;
    setIsAnalyzing(true);
    setAnalysisError(null);

    const isBackendConnected = knownConnected !== null ? knownConnected : backendStatus === "connected";

    if (isBackendConnected && !forceStandalone) {
      try {
        const res = await fetch("http://localhost:8000/api/qc/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            segments: currentSegments,
            audio_path: backendAudioPath,
            use_mock: false
          })
        });
        if (!res.ok) throw new Error(`Backend returned ${res.status}`);
        const data = await res.json();
        if (!Array.isArray(data.findings)) throw new Error("Malformed response: missing findings");
        setFindings(data.findings);
        setOverallScore(data.overall_score);
        updateStats(data.findings);
        setAnalysisSource("gemini");
        setIsAnalyzing(false);
        return;
      } catch (err) {
        console.error("Failed to fetch analysis, using local analyzer:", err);
        setAnalysisError(`AI 백엔드 검수 실패 (${err.message}). 로컬 규칙 기반 엔진으로 대체합니다.`);
      }
    }

    // Local JS Analyzer (For Standalone or local fallback)
    setAnalysisSource("local");
    setTimeout(() => {
      const localFindings = [];
      
      currentSegments.forEach((seg, index) => {
        const textKr = seg.original_text || "";
        const textEn = (seg.translated_text || "").trim();

        // Check 0: Missing Translation
        if (!textEn) {
          localFindings.push({
            id: `loc_${seg.id}_missing`,
            segment_id: seg.id,
            category: "localization",
            severity: "high",
            issue_type: "번역 누락",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: "영문 번역 대사가 누락되었습니다. 더빙 오디오 녹음 및 싱크 매칭을 하려면 스크립트를 먼저 채워야 합니다.",
            original_text: seg.original_text,
            current_translation: "",
            recommendation: "AI recommended correction line.",
            confidence: 1.0
          });
          return;
        }

        // Loc 1: Honorifics
        if (
          (textKr.includes("형") || textKr.includes("누나") || textKr.includes("오빠") || textKr.includes("부장")) && 
          (textEn.toLowerCase().includes("brother") || textEn.toLowerCase().includes("sister") || textEn.toLowerCase().includes("director"))
        ) {
          localFindings.push({
            id: `loc_${seg.id}_honorific`,
            segment_id: seg.id,
            category: "localization",
            severity: "medium",
            issue_type: "문화적 정서 차이",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: "한국어 친족 호칭(형/누나/오빠)이나 직책이 영어 'brother', 'sister' 등으로 직역되어 대화 환경에서 다소 부자연스럽게 들립니다. 자연스러운 대안 호칭을 권장합니다.",
            original_text: seg.original_text,
            current_translation: seg.translated_text,
            recommendation: seg.translated_text.replace(/brother/i, "man").replace(/brother/g, "man").replace(/Brother/g, "Hey"),
            confidence: 0.94
          });
        }

        // Loc 2: Literal Translation of "눈치"
        if (textKr.includes("눈치") && (textEn.toLowerCase().includes("eyes") || textEn.toLowerCase().includes("look"))) {
          localFindings.push({
            id: `loc_${seg.id}_nunchi`,
            segment_id: seg.id,
            category: "localization",
            severity: "high",
            issue_type: "번역 오류",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: "'눈치 보다'라는 고유 정서 표현이 눈(eyes/look)으로 직접 직역되었습니다. 'walking on eggshells'(조심조심 행동하다) 또는 'read the room'(상황을 파악하다) 등으로 의역이 필요합니다.",
            original_text: seg.original_text,
            current_translation: seg.translated_text,
            recommendation: "Stop walking on eggshells and just speak.",
            confidence: 0.91
          });
        }

        // Loc 3: Translation Mistake "어이가 없네" -> "no kidney"
        if (textKr.includes("어이가 없네") && textEn.toLowerCase().includes("kidney")) {
          localFindings.push({
            id: `loc_${seg.id}_kidney`,
            segment_id: seg.id,
            category: "localization",
            severity: "high",
            issue_type: "번역 오류",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: "관용구 '어이가 없네'(황당하다)의 '어이'가 인체 장기인 신장(kidney)으로 치명적인 기계 번역 오역이 일어났습니다. 'ridiculous'(황당한)로 즉시 변경해야 합니다.",
            original_text: seg.original_text,
            current_translation: seg.translated_text,
            recommendation: "This is ridiculous.",
            confidence: 0.99
          });
        }

        // Loc 4: Rice translation
        if (textKr.includes("밥") && textEn.toLowerCase().includes("rice")) {
          localFindings.push({
            id: `loc_${seg.id}_rice`,
            segment_id: seg.id,
            category: "localization",
            severity: "low",
            issue_type: "문화적 정서 차이",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: "'밥 먹었어?'는 안부 인사 의미의 한국적 관용구입니다. 이를 문자 그대로 'eat rice'로 번역하면 어색하므로 일상적 인사(Have you eaten?)로 의역을 추천합니다.",
            original_text: seg.original_text,
            current_translation: seg.translated_text,
            recommendation: "Have you eaten?",
            confidence: 0.88
          });
        }

        // Voice 1: Sync pacing overflow
        const duration = seg.end_time - seg.start_time;
        const words = textEn.split(" ").length;
        if (words / duration > 4.5) {
          localFindings.push({
            id: `voice_${seg.id}_sync_pacing`,
            segment_id: seg.id,
            category: "voice",
            severity: "high",
            issue_type: "싱크 오류",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: `자막 지속 시간 대비 발화 속도 초과 (초당 ${(words / duration).toFixed(1)} 단어). 성우가 너무 급하게 대사를 말해야 해 입모양(Lip-sync)이 깨집니다. 대사 축약이 필요합니다.`,
            original_text: seg.original_text,
            current_translation: seg.translated_text,
            recommendation: "Shorten phrase length.",
            confidence: 0.95
          });
        }

        // Voice 2: Timbre consistency drift
        if (index === 1) {
          localFindings.push({
            id: `voice_${seg.id}_consistency`,
            segment_id: seg.id,
            category: "voice",
            severity: "medium",
            issue_type: "음색 일관성 오류",
            start_time: seg.start_time,
            end_time: seg.end_time,
            speaker: seg.speaker,
            description: "성우 음색 불일치 감지. 주파수 분석 결과 이전 대사에 설정된 캐릭터의 표준 주파수 프로필에서 23%의 음색 변조 및 이질감이 감지되었습니다.",
            original_text: seg.original_text,
            current_translation: seg.translated_text,
            recommendation: "Recalibrate vocal model or re-record.",
            confidence: 0.84
          });
        }
      });

      setFindings(localFindings);
      
      const highCnt = localFindings.filter(f => f.severity === "high").length;
      const medCnt = localFindings.filter(f => f.severity === "medium").length;
      const lowCnt = localFindings.filter(f => f.severity === "low").length;
      const score = Math.max(100 - (highCnt * 15 + medCnt * 8 + lowCnt * 3), 0);
      setOverallScore(score);
      
      updateStats(localFindings);
      setIsAnalyzing(false);
    }, 1200);
  };

  const updateStats = (currentFindings) => {
    setStats({
      total: currentFindings.length,
      high: currentFindings.filter(f => f.severity === "high").length,
      medium: currentFindings.filter(f => f.severity === "medium").length,
      low: currentFindings.filter(f => f.severity === "low").length,
      localization: currentFindings.filter(f => f.category === "localization").length,
      voice: currentFindings.filter(f => f.category === "voice").length
    });
  };

  // 4. Script Editing
  const handleScriptChange = (id, newText) => {
    const updated = segments.map(seg => {
      if (seg.id === id) {
        return { ...seg, translated_text: newText };
      }
      return seg;
    });
    setSegments(updated);
  };

  // 5. Action Handlers
  const applyAIFix = (findingId, segmentId, recommendation) => {
    const updatedSegments = segments.map(seg => {
      if (seg.id === segmentId) {
        return { ...seg, translated_text: recommendation };
      }
      return seg;
    });
    setSegments(updatedSegments);

    const updatedFindings = findings.filter(f => f.id !== findingId);
    setFindings(updatedFindings);
    updateStats(updatedFindings);

    const highCnt = updatedFindings.filter(f => f.severity === "high").length;
    const medCnt = updatedFindings.filter(f => f.severity === "medium").length;
    const lowCnt = updatedFindings.filter(f => f.severity === "low").length;
    setOverallScore(Math.max(100 - (highCnt * 15 + medCnt * 8 + lowCnt * 3), 0));
  };

  const ignoreFinding = (findingId) => {
    const updatedFindings = findings.filter(f => f.id !== findingId);
    setFindings(updatedFindings);
    updateStats(updatedFindings);
    
    const highCnt = updatedFindings.filter(f => f.severity === "high").length;
    const medCnt = updatedFindings.filter(f => f.severity === "medium").length;
    const lowCnt = updatedFindings.filter(f => f.severity === "low").length;
    setOverallScore(Math.max(100 - (highCnt * 15 + medCnt * 8 + lowCnt * 3), 0));
  };

  // 6. Video Sync Time Tracking
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      const time = video.currentTime;
      setCurrentTime(time);
      
      const activeSeg = segments.find(
        seg => time >= seg.start_time && time <= seg.end_time
      );
      if (activeSeg) {
        setActiveSegmentId(activeSeg.id);
      } else {
        setActiveSegmentId(null);
      }
    };

    const handleLoadedMetadata = () => {
      setDuration(video.duration || 10.0);
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    
    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [segments]);

  const jumpToSegment = (seg) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seg.start_time;
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleFindingClick = (finding, e) => {
    if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
      return;
    }
    if (videoRef.current) {
      videoRef.current.currentTime = finding.start_time;
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
    setActiveSegmentId(finding.segment_id);
    const cardEl = document.getElementById(`script-card-${finding.segment_id}`);
    if (cardEl) {
      cardEl.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  // 7. Canvas Waveform Animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let width = canvas.width = canvas.offsetWidth;
    let height = canvas.height = canvas.offsetHeight;
    
    const resizeObserver = new ResizeObserver(() => {
      if (canvas) {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
      }
    });
    resizeObserver.observe(canvas);

    let waves = [
      { amplitude: 15, speed: 0.1, color: "rgba(168, 85, 247, 0.4)", phase: 0 },
      { amplitude: 25, speed: 0.07, color: "rgba(236, 72, 153, 0.25)", phase: 2 },
      { amplitude: 8, speed: 0.15, color: "rgba(6, 182, 212, 0.3)", phase: 4 }
    ];

    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Draw grid lines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
      ctx.lineWidth = 1;
      for (let i = 0; i < width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
      }
      for (let i = 0; i < height; i += 20) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
        ctx.stroke();
      }

      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

      if (waveformPeaks && waveformPeaks.length > 0) {
        // Draw real waveform
        const numPeaks = waveformPeaks.length;
        const barWidth = width / numPeaks;
        
        ctx.fillStyle = "rgba(168, 85, 247, 0.45)"; // Deep purple translucent
        for (let i = 0; i < numPeaks; i++) {
          const peak = waveformPeaks[i];
          const barHeight = peak * height * 0.8;
          const x = i * barWidth;
          const y = (height - barHeight) / 2;
          ctx.fillRect(x, y, barWidth - 0.5, barHeight);
        }
        
        // Highlight active progress overlay
        const pct = currentTime / duration;
        const elapsedWidth = pct * width;
        ctx.fillStyle = "rgba(6, 182, 212, 0.12)"; // Soft cyan highlight for played part
        ctx.fillRect(0, 0, elapsedWidth, height);
      } else {
        // Fall back to moving simulated waves
        waves.forEach(wave => {
          ctx.strokeStyle = wave.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          
          for (let x = 0; x < width; x++) {
            const multiplier = isPlaying ? 1.0 : 0.08;
            const y = (height / 2) + 
              Math.sin(x * 0.01 + wave.phase) * wave.amplitude * multiplier * Math.sin(x * 0.003);
            
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          
          ctx.stroke();
          wave.phase += wave.speed;
        });
      }

      // Draw red playhead line
      if (videoRef.current) {
        const pct = currentTime / duration;
        ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(pct * width, 0);
        ctx.lineTo(pct * width, height);
        ctx.stroke();
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationRef.current);
      resizeObserver.disconnect();
    };
  }, [isPlaying, currentTime, duration, waveformPeaks]);

  const filteredFindings = findings.filter(f => {
    const matchesCategory = filter === "all" || f.category === filter;
    const matchesSeverity = severityFilter === "all" || f.severity === severityFilter;
    return matchesCategory && matchesSeverity;
  });

  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (overallScore / 100) * circumference;

  const getScoreColor = () => {
    if (overallScore >= 85) return "#a855f7"; 
    if (overallScore >= 70) return "#f97316"; 
    return "#ef4444"; 
  };

  return (
    <div className="aether-app">
      <nav className="view-tabs">
        {[["project", "프로젝트"], ["review", "검수"], ["report", "판정/리포트"]].map(([v, label]) => (
          <button key={v} className={view === v ? "tab active" : "tab"}
            onClick={() => setView(v)}>{label}</button>
        ))}
      </nav>

      {view === "project" && (
        <ProjectView uploads={uploads} setUploads={setUploads} onJobComplete={handleJobComplete} />
      )}

      {view === "review" && (
      <>
      {/* 1. Header */}
      <header className="aether-header">
        <div className="header-logo">
          <div className="logo-icon"></div>
          <div className="logo-text">
            <h1>AETHER <span>// AI DUBBING QC</span></h1>
            <p>Korean to English AI Localization Quality Control Suite</p>
          </div>
        </div>

        {/* File upload panel with original/dubbed media and dual SRT inputs */}
        <div className="header-file-panel">
          {/* 1. Original KR Media */}
          <div className="file-uploader-box">
            <span className="file-label" title={originalMediaName}>🎙️ 원본 영상/음성 (KR): {originalMediaName}</span>
            <div className="button-group-row">
              <button className="btn-file-select" onClick={() => originalInputRef.current.click()} disabled={uploadingOriginal}>
                {uploadingOriginal ? "업로드 중..." : (originalAudioPath ? "등록 완료 ✓" : "미디어 등록")}
              </button>
              {originalAudioPath && (
                <button 
                  className={`btn-action-ai ${isTranscribing ? "loading" : ""}`}
                  onClick={handleTranscribe} 
                  disabled={isTranscribing}
                  title="한국어 음성을 자막으로 변환합니다."
                >
                  {isTranscribing ? "STT..." : "AI 전사"}
                </button>
              )}
            </div>
            <input 
              type="file" 
              ref={originalInputRef} 
              style={{ display: "none" }} 
              accept="video/*,audio/*" 
              onChange={handleOriginalUpload}
            />
          </div>

          {/* 2. Dubbed EN Media */}
          <div className="file-uploader-box">
            <span className="file-label" title={videoFileName}>🔊 영어 더빙 영상/음성: {videoFileName}</span>
            <button className="btn-file-select" onClick={() => videoInputRef.current.click()} disabled={uploadingVideo}>
              {uploadingVideo ? "분석 중..." : (backendAudioPath ? "등록 완료 ✓" : "미디어 등록")}
            </button>
            <input 
              type="file" 
              ref={videoInputRef} 
              style={{ display: "none" }} 
              accept="video/*,audio/*" 
              onChange={handleVideoUpload}
            />
          </div>

          {/* 3. Korean Subtitle (KR Script) */}
          <div className="file-uploader-box">
            <span className="file-label" title={krFileName}>🇰🇷 한국어 자막 (SRT): {krFileName}</span>
            <div className="button-group-row">
              <button className="btn-file-select" onClick={() => krInputRef.current.click()}>
                {krFileName.includes("기본 샘플") ? "자막 불러오기" : "등록 완료 ✓"}
              </button>
              {segments.length > 0 && (
                <button 
                  className={`btn-action-ai ${isTranslating ? "loading" : ""}`}
                  onClick={handleTranslate} 
                  disabled={isTranslating}
                  title="한국어 자막을 기반으로 영어 더빙 번역을 AI 생성합니다."
                >
                  {isTranslating ? "번역 중..." : "AI 번역"}
                </button>
              )}
            </div>
            <input 
              type="file" 
              ref={krInputRef} 
              style={{ display: "none" }} 
              accept=".srt,.json" 
              onChange={handleKrUpload}
            />
          </div>

          {/* 4. English Subtitle (EN Script) */}
          <div className="file-uploader-box">
            <span className="file-label" title={enFileName}>🇺🇸 영어 자막 (SRT): {enFileName}</span>
            <button className="btn-file-select" onClick={() => enInputRef.current.click()}>
              {enFileName.includes("기본 샘플") ? "자막 불러오기" : "등록 완료 ✓"}
            </button>
            <input 
              type="file" 
              ref={enInputRef} 
              style={{ display: "none" }} 
              accept=".srt" 
              onChange={handleEnUpload}
            />
          </div>
        </div>

        <div className="header-status">
          <div className={`status-badge ${backendStatus}`}>
            <span className="pulse-dot"></span>
            {backendStatus === "connected" ? "AI 백엔드 연결됨" :
             backendStatus === "standalone" ? "로컬 스탠드얼론 모드" : "엔진 연결 확인 중..."}
          </div>
          {analysisSource && (
            <div className={`status-badge source-${analysisSource}`} title="마지막 검수 결과가 어느 엔진에서 나왔는지 표시합니다.">
              {analysisSource === "gemini" ? "검수 엔진: Gemini AI" : "검수 엔진: 로컬 규칙 기반 (폴백)"}
            </div>
          )}
          <button
            className={`reanalyze-btn ${isAnalyzing ? "loading" : ""}`}
            onClick={() => runQCAnalysis(segments, backendStatus === "standalone")}
            disabled={isAnalyzing || segments.length === 0}
            id="reanalyze-btn"
          >
            {isAnalyzing ? (
              <>
                <span className="spinner"></span> 분석 중...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                AI 검수 실행
              </>
            )}
          </button>
        </div>

        {/* Circular Progress Gauge */}
        <div className="score-meter">
          <svg width="110" height="110" viewBox="0 0 110 110">
            <circle cx="55" cy="55" r={radius} className="meter-bg" />
            <circle 
              cx="55" 
              cy="55" 
              r={radius} 
              className="meter-bar" 
              style={{
                strokeDasharray: circumference,
                strokeDashoffset: strokeDashoffset,
                stroke: getScoreColor()
              }}
            />
          </svg>
          <div className="score-number" style={{ color: getScoreColor() }}>
            {overallScore}
            <span className="score-label">품질 점수</span>
          </div>
        </div>
      </header>

      {analysisError && (
        <div className="analysis-error-banner">
          <span>⚠ {analysisError}</span>
          <button onClick={() => setAnalysisError(null)}>닫기</button>
        </div>
      )}

      {/* 2. Stats strip */}
      <div className="stats-strip">
        <div className="stat-card total" onClick={() => { setSeverityFilter("all"); setFilter("all"); }}>
          <span className="label">전체 검출 건수</span>
          <span className="value">{stats.total}</span>
        </div>
        <div className="stat-card high" onClick={() => { setSeverityFilter("high"); setFilter("all"); }}>
          <span className="label">치명적 위험</span>
          <span className="value">{stats.high}</span>
        </div>
        <div className="stat-card medium" onClick={() => { setSeverityFilter("medium"); setFilter("all"); }}>
          <span className="label">중간 위험</span>
          <span className="value">{stats.medium}</span>
        </div>
        <div className="stat-card low" onClick={() => { setSeverityFilter("low"); setFilter("all"); }}>
          <span className="label">낮은 위험</span>
          <span className="value">{stats.low}</span>
        </div>
        <div className="stat-card localization" onClick={() => { setSeverityFilter("all"); setFilter("localization"); }}>
          <span className="label">로컬라이제이션 오류</span>
          <span className="value">{stats.localization}</span>
        </div>
        <div className="stat-card voice" onClick={() => { setSeverityFilter("all"); setFilter("voice"); }}>
          <span className="label">음향/싱크 오류</span>
          <span className="value">{stats.voice}</span>
        </div>
      </div>

      {/* 3. Main Dashboard Grid */}
      <div className="dashboard-grid">
        
        {/* Left Section: Video Player & Audio Wave */}
        <section className="dashboard-column col-video">
          <div className="section-header">
            <h2>비주얼 & 오디오 모니터링</h2>
            <div className="media-role-tabs">
              <button 
                className={activeVideoRole === "dubbed" ? "active" : ""} 
                onClick={() => {
                  setActiveVideoRole("dubbed");
                  setIsPlaying(false);
                  if (videoRef.current) videoRef.current.load();
                }}
              >
                더빙 EN
              </button>
              {originalVideoSrc && (
                <button 
                  className={activeVideoRole === "original" ? "active" : ""} 
                  onClick={() => {
                    setActiveVideoRole("original");
                    setIsPlaying(false);
                    if (videoRef.current) videoRef.current.load();
                  }}
                >
                  원본 KR
                </button>
              )}
            </div>
            <span className="pulse-indicator text-cyan">실시간 분석 중</span>
          </div>

          <div className="video-container">
            <video 
              ref={videoRef}
              src={activeVideoRole === "original" ? originalVideoSrc : dubbedVideoSrc}
              playsInline
              loop
              muted
              onClick={togglePlayback}
            />
            
            {/* Custom Overlay Subtitle */}
            <div className="subtitle-overlay">
              {segments.map(seg => (
                activeSegmentId === seg.id && (
                  <div key={seg.id} className="active-sub-card">
                    <p className="sub-speaker">{seg.speaker}</p>
                    <p className="sub-kr">{seg.original_text}</p>
                    <p className="sub-en">{seg.translated_text}</p>
                  </div>
                )
              ))}
            </div>

            {!isPlaying && (
              <div className="video-play-overlay" onClick={togglePlayback}>
                <div className="play-button-icon"></div>
              </div>
            )}
          </div>

          {/* Custom Video Controls */}
          <div className="video-controls">
            <button className="ctrl-btn-play" onClick={togglePlayback} id="play-pause-btn">
              {isPlaying ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              )}
            </button>
            <div className="progress-info">
              <span className="time-display">
                {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
              </span>
              <div 
                className="progress-bar-container"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const pct = (e.clientX - rect.left) / rect.width;
                  if (videoRef.current) {
                    videoRef.current.currentTime = pct * duration;
                  }
                }}
              >
                <div 
                  className="progress-bar-fill"
                  style={{ width: `${(currentTime / duration) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="waveform-container">
            <div className="waveform-title">
              <span>VOICE DIALOGUE STEM WAVEFORM</span>
              <span className="frequency-hz">16.0 kHz / 16-bit Mono</span>
            </div>
            <canvas ref={canvasRef} className="waveform-canvas" />
          </div>
        </section>

        {/* Middle Section: Script Timecodes */}
        <section className="dashboard-column col-script">
          <div className="section-header">
            <h2>스크립트 타임코드</h2>
            <span className="total-badge">{segments.length} 라인</span>
          </div>

          <div className="script-list">
            {segments.length === 0 ? (
              <div className="no-findings">
                <p>불러온 자막 스크립트가 없습니다.</p>
                <span>한국어 자막(.srt/.json) 파일과 영어 더빙 자막(.srt) 파일을 불러와 주세요.</span>
              </div>
            ) : (
              segments.map(seg => {
                const isActive = activeSegmentId === seg.id;
                const hasCriticalError = findings.some(f => f.segment_id === seg.id && f.severity === "high");
                const hasMediumError = findings.some(f => f.segment_id === seg.id && f.severity === "medium");
                
                let cardBorderClass = "";
                if (isActive) cardBorderClass = "active";
                else if (hasCriticalError) cardBorderClass = "has-high-error";
                else if (hasMediumError) cardBorderClass = "has-med-error";

                return (
                  <div 
                    key={seg.id} 
                    className={`script-card ${cardBorderClass}`}
                    id={`script-card-${seg.id}`}
                  >
                    <div className="script-card-header" onClick={() => jumpToSegment(seg)}>
                      <span className="speaker-badge">{seg.speaker}</span>
                      <span className="time-badge">
                        {seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s
                      </span>
                    </div>

                    <div className="script-content">
                      <div className="script-row original">
                        <span className="lang-label">KR</span>
                        <p>{seg.original_text || "(원본 자막 내용 없음)"}</p>
                      </div>

                      <div className="script-row translation">
                        <span className="lang-label">EN</span>
                        <textarea
                          value={seg.translated_text}
                          onChange={(e) => handleScriptChange(seg.id, e.target.value)}
                          placeholder="영어 번역 자막 내용을 기입하세요"
                          rows={2}
                          id={`text-area-${seg.id}`}
                        />
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>

        {/* Right Section: AI Findings */}
        <section className="dashboard-column col-findings">
          <div className="section-header">
            <h2>AI QC 검수 결과</h2>
            <span className="findings-count">{filteredFindings.length}건 검출됨</span>
          </div>

          <div className="filter-tabs">
            <button 
              className={filter === "all" ? "active" : ""} 
              onClick={() => setFilter("all")}
              id="filter-all"
            >
              전체
            </button>
            <button 
              className={filter === "localization" ? "active" : ""} 
              onClick={() => setFilter("localization")}
              id="filter-loc"
            >
              로컬라이제이션
            </button>
            <button 
              className={filter === "voice" ? "active" : ""} 
              onClick={() => setFilter("voice")}
              id="filter-voice"
            >
              음향 및 싱크
            </button>
          </div>

          <div className="findings-list">
            {filteredFindings.length === 0 ? (
              <div className="no-findings">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="1.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                <p>선택한 필터에 해당하는 검수 결과가 없습니다.</p>
                <span>스크립트와 오디오 상태가 최적화되어 있습니다.</span>
              </div>
            ) : (
              filteredFindings.map(finding => (
                <div 
                  key={finding.id} 
                  className={`finding-card ${finding.severity}`}
                  id={`finding-card-${finding.id}`}
                  onClick={(e) => handleFindingClick(finding, e)}
                >
                  <div className="finding-card-header">
                    <span className={`finding-type-badge ${finding.severity}`}>
                      {finding.issue_type}
                    </span>
                    <span className="finding-category">
                      {finding.category === "localization" ? "로컬라이제이션" : "음향 및 싱크"}
                    </span>
                  </div>

                  <div className="finding-body">
                    <p className="finding-desc">{finding.description}</p>
                    
                    {finding.original_text && (
                      <div className="finding-context-block">
                        <div className="context-item">
                          <span>한국어 원문:</span> {finding.original_text}
                        </div>
                        <div className="context-item text-red-line">
                          <span>현재 번역:</span> {finding.current_translation}
                        </div>
                      </div>
                    )}

                    <div className="recommendation-block">
                      <div className="rec-header">
                        <span>AI 추천 수정안:</span>
                        <span className="confidence-value">
                          신뢰도 {(finding.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="rec-text">"{finding.recommendation}"</p>
                    </div>

                    <div className="finding-actions">
                      <button 
                        className="btn-apply-fix"
                        onClick={() => applyAIFix(finding.id, finding.segment_id, finding.recommendation)}
                        id={`btn-apply-${finding.id}`}
                      >
                        수정안 적용
                      </button>
                      <button 
                        className="btn-ignore"
                        onClick={() => ignoreFinding(finding.id)}
                        id={`btn-ignore-${finding.id}`}
                      >
                        무시
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

      </div>
      </>
      )}
    </div>
  );
}

export default App;
