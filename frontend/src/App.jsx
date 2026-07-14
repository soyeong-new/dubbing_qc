import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import ProjectView from "./views/ProjectView";
import ReportView from "./views/ReportView";
import { postFeedback } from "./api";

function App() {
  const [view, setView] = useState("project"); // "project" | "review" | "report"
  const [uploads, setUploads] = useState({});
  const [jobId, setJobId] = useState(null);
  const [qcResult, setQcResult] = useState(null);
  const [reviewedFindings, setReviewedFindings] = useState({}); // {findingId: {action, finalText}}
  const [pendingReviews, setPendingReviews] = useState({}); // {findingId: true} while POST in flight
  const [reviewErrors, setReviewErrors] = useState({}); // {findingId: errorMessage}
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

  // Real waveform & backend audio path states
  const [waveformPeaks, setWaveformPeaks] = useState([]);
  const [backendAudioPath, setBackendAudioPath] = useState(null);
  const [uploadingVideo, setUploadingVideo] = useState(false);

  // Original media (Review 탭에서 원본 재생용 — QC 실행과는 무관)
  const [originalMediaName, setOriginalMediaName] = useState("선택되지 않음");
  const [originalAudioPath, setOriginalAudioPath] = useState(null);
  const [uploadingOriginal, setUploadingOriginal] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const videoInputRef = useRef(null);
  const originalInputRef = useRef(null);

  // 1. Job completion handler — 백엔드 QC 잡의 결과(QCResult)를 검수/리포트 뷰
  // 상태로 반영한다. 여기서 옮겨지는 findings/segments는 실제 백엔드 파이프라인
  // 산출물이며, 클라이언트 측에서 새로 생성/변형되지 않는다.
  const handleJobComplete = (id, result, movieTitle) => {
    setJobId(id);
    setQcResult({ ...result, movie_title: movieTitle || "untitled" });
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

  const updateStats = (currentFindings) => {
    const highCnt = currentFindings.filter(f => f.severity === "high").length;
    const medCnt = currentFindings.filter(f => f.severity === "medium").length;
    const lowCnt = currentFindings.filter(f => f.severity === "low").length;
    setStats({
      total: currentFindings.length,
      high: highCnt,
      medium: medCnt,
      low: lowCnt,
      localization: currentFindings.filter(f => f.category === "localization").length,
      voice: currentFindings.filter(f => f.category === "voice").length
    });
    // 게이지 표시용 참고 점수 — 공식 판정은 Report 탭의 5축 MOS/판정을 따른다
    setOverallScore(Math.max(100 - (highCnt * 15 + medCnt * 8 + lowCnt * 3), 0));
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
  // 검수 액션: 모든 클릭이 피드백 저장소에 기록된다 (학습 데이터 축적 입구)
  const reviewFinding = async (finding, action, finalText = "", chosenPersona = "") => {
    // 이미 확정됐거나 요청이 진행 중이면 중복 제출을 막는다
    if (reviewedFindings[finding.id] || pendingReviews[finding.id]) return;
    setPendingReviews((p) => ({ ...p, [finding.id]: true }));
    setReviewErrors((e) => {
      const next = { ...e };
      delete next[finding.id];
      return next;
    });
    try {
      await postFeedback({
        movie: qcResult?.movie_title || "untitled",
        segment_id: finding.segment_id,
        korean: finding.original_text,
        dubbed: finding.current_translation,
        finding_id: finding.id,
        reviewer_action: action, // "approved" | "rejected" | "modified"
        final_text: finalText,
        chosen_persona: chosenPersona,
      });
      setReviewedFindings((r) => ({ ...r, [finding.id]: { action, finalText } }));
    } catch (err) {
      setReviewErrors((e) => ({ ...e, [finding.id]: "저장 실패 — 다시 시도해주세요." }));
    } finally {
      setPendingReviews((p) => {
        const next = { ...p };
        delete next[finding.id];
        return next;
      });
    }
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

        {/* 재생용 미디어 등록 (Review 탭 전용 — QC 실행은 프로젝트 탭에서 이미 완료된 잡의 결과다) */}
        <div className="header-file-panel">
          {/* 1. Original KR Media */}
          <div className="file-uploader-box">
            <span className="file-label" title={originalMediaName}>🎙️ 원본 영상/음성 (KR): {originalMediaName}</span>
            <button className="btn-file-select" onClick={() => originalInputRef.current.click()} disabled={uploadingOriginal}>
              {uploadingOriginal ? "업로드 중..." : (originalAudioPath ? "등록 완료 ✓" : "미디어 등록")}
            </button>
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

                    <div className="persona-alternatives">
                      {Object.entries(finding.alternatives || {}).map(([persona, suggestion]) => (
                        <button key={persona} className="alt-chip"
                          title={`${persona}의 수정안 채택`}
                          disabled={!!pendingReviews[finding.id] || !!reviewedFindings[finding.id]}
                          onClick={() => reviewFinding(finding, "modified", suggestion, persona)}>
                          <span className="alt-persona">{persona}</span>
                          <span className="alt-text">{suggestion}</span>
                        </button>
                      ))}
                    </div>
                    <div className="finding-meta">
                      동의 {finding.agreement}/3 · {finding.axis} · {finding.source === "rule" ? "룰 체크" : finding.source.replace("persona:", "")}
                    </div>
                    {reviewErrors[finding.id] && (
                      <div className="review-error">{reviewErrors[finding.id]}</div>
                    )}
                    <div className="review-actions">
                      {reviewedFindings[finding.id] ? (
                        <span className="reviewed-badge">
                          {reviewedFindings[finding.id].action === "approved" ? "✓ 승인됨"
                            : reviewedFindings[finding.id].action === "rejected" ? "✕ 반려됨(오탐)"
                            : "✎ 수정 확정"}
                        </span>
                      ) : (
                        <>
                          <button className="btn-approve" disabled={!!pendingReviews[finding.id]}
                            onClick={() => reviewFinding(finding, "approved")}>승인</button>
                          <button className="btn-reject" disabled={!!pendingReviews[finding.id]}
                            onClick={() => reviewFinding(finding, "rejected")}>반려 (오탐)</button>
                          <button className="btn-modify" disabled={!!pendingReviews[finding.id]} onClick={() => {
                            const text = window.prompt("최종 영어 대사를 입력하세요:", finding.recommendation);
                            if (text) reviewFinding(finding, "modified", text);
                          }}>직접 수정</button>
                        </>
                      )}
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

      {view === "report" && (
        <ReportView result={qcResult} jobId={jobId}
          findings={findings} reviewed={reviewedFindings} />
      )}
    </div>
  );
}

export default App;
