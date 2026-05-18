import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

const isMobile = new URLSearchParams(window.location.search).get('platform') === 'mobile';

const POSES = [
  { value: 'squat', label: 'Squat' },
  { value: 'plank', label: 'Plank' },
  { value: 'downdog', label: 'Downward Dog' },
  { value: 'tree', label: 'Tree' },
  { value: 'warrior2', label: 'Warrior II' },
  { value: 'goddess', label: 'Goddess' },
  { value: 'sitting', label: 'Sitting' }
];

const INTERVAL_OPTIONS = [
  { value: 700, label: 'Fast (0.7s)' },
  { value: 1200, label: 'Balanced (1.2s)' },
  { value: 1800, label: 'Light (1.8s)' }
];

const VOICE_COOLDOWN_MS = 6500;
const VOICE_COOLDOWN_GOOD_MS = 15000; // positive cues less frequent
const STABILITY_WINDOW = 5;
const STABILITY_REQUIRED = 3;

function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function formatMetricValue(value, suffix = '') {
  return value === undefined || value === null ? '-' : `${value}${suffix}`;
}

function buildMetricRows(feedback, selectedPose, lastUpdated) {
  const metrics = feedback.metrics || {};

  if (selectedPose === 'sitting') {
    return [
      ['Neck Tilt', formatMetricValue(metrics.neck_tilt, ' ratio')],
      ['Shoulder Level', formatMetricValue(metrics.shoulder_tilt, ' ratio')],
      ['Spine Lean', formatMetricValue(metrics.torso_lean, ' ratio')],
      ['Spine Extension', formatMetricValue(metrics.spine_extension, ' ratio')],
      ['Side', metrics.side ?? '-'],
      ['Visibility', feedback.visibility?.score ?? '-'],
      ['Issue', feedback.issue ?? feedback.error ?? '-'],
      ['Source', feedback.source ?? '-'],
      ['Updated', lastUpdated || '-'],
    ];
  }

  const rows = [
    ['Knee Angle', formatMetricValue(metrics.knee_angle, ' deg')],
    ['Hip Angle', formatMetricValue(metrics.hip_angle, ' deg')],
    ['Torso Lean', metrics.torso_lean ?? '-'],
    ['Side', metrics.side ?? '-'],
    ['Visibility', feedback.visibility?.score ?? '-'],
    ['Issue', feedback.issue ?? feedback.error ?? '-'],
  ];

  if (feedback.phase) {
    rows.push(['Phase', feedback.phase]);
  }

  if (feedback.source) {
    rows.push(['Source', feedback.source]);
  }

  rows.push(['Updated', lastUpdated || '-']);
  return rows;
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const canvasCtxRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const coachSectionRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const selectedPoseRef = useRef('squat');
  const lastSpokenRef = useRef({ text: '', at: 0 });
  const voiceEnabledRef = useRef(false);
  const isAnalyzingRef = useRef(false);
  const feedbackWindowRef = useRef([]);
  const squatTrackerRef = useRef({ seenBottom: false, lastPhase: '', lastRepAt: 0 });
  const runtimeStatsRef = useRef({ count: 0, success: 0, avgLatency: 0 });
  const feedbackRef = useRef(null);
  const metricsRef = useRef(null);

  const [selectedPose, setSelectedPose] = useState('squat');
  const [intervalMs, setIntervalMs] = useState(1200);
  const [showCoach, setShowCoach] = useState(false);
  const [isRealtime, setIsRealtime] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  // keep a ref so analyzeFrame (non-reactive) can read it without stale closure
  useEffect(() => { voiceEnabledRef.current = voiceEnabled; }, [voiceEnabled]);
  const facingModeRef = useRef('environment'); // start with rear camera
  const [cameraPermission, setCameraPermission] = useState('unknown');
  const [cameraHint, setCameraHint] = useState('Click "Start Live Coach" to enable camera.');
  const [history, setHistory] = useState([]);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [successCount, setSuccessCount] = useState(0);
  const [avgLatencyMs, setAvgLatencyMs] = useState(0);
  const [lastLatencyMs, setLastLatencyMs] = useState(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [lastUpdated, setLastUpdated] = useState('');
  const [stabilityMessage, setStabilityMessage] = useState('');
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const [poseSeconds, setPoseSeconds] = useState(0);
  const [squatReps, setSquatReps] = useState(0);
  const [poseDetected, setPoseDetected] = useState(false);

  useEffect(() => {
    selectedPoseRef.current = selectedPose;
  }, [selectedPose]);

  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
        });
        if (!cancelled) {
          poseLandmarkerRef.current = landmarker;
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Failed to initialize MediaPipe: ${err.message}`);
        }
      }
    }

    initMediaPipe();

    return () => {
      cancelled = true;
      poseLandmarkerRef.current?.close?.();
      poseLandmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cached = localStorage.getItem('ygb-history');
    if (!cached) {
      return;
    }
    try {
      const parsed = JSON.parse(cached);
      if (Array.isArray(parsed)) {
        setHistory(parsed.slice(0, 25));
      }
    } catch {
      localStorage.removeItem('ygb-history');
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('ygb-history', JSON.stringify(history.slice(0, 25)));
  }, [history]);

  useEffect(() => {
    let cancelled = false;

    async function readCameraPermission() {
      if (!navigator.permissions?.query) {
        return;
      }
      try {
        const result = await navigator.permissions.query({ name: 'camera' });
        if (!cancelled) {
          setCameraPermission(result.state);
          if (result.state === 'denied') {
            setCameraHint('Camera permission is denied. Allow it from browser site settings and retry.');
          }
        }
        result.onchange = () => {
          setCameraPermission(result.state);
        };
      } catch {
        // Some browsers do not support camera permissions query.
      }
    }

    async function requestCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('Camera API is not available in this browser. Use Chrome/Safari/Edge on secure origin.');
        setCameraHint('Camera API unavailable.');
        return;
      }
      try {
        await startCamera(facingModeRef.current, cancelled);
      } catch {
        if (!cancelled) {
          setCameraReady(false);
          setCameraPermission('denied');
          setError('Unable to access camera. Please allow webcam permissions.');
          setCameraHint('Permission blocked. Open browser site permissions and enable camera access.');
        }
      }
    }

    readCameraPermission();

    return () => {
      cancelled = true;
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  async function startCamera(mode = facingModeRef.current, cancelled = false) {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((t) => t.stop());
      cameraStreamRef.current = null;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: mode, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    if (!cancelled && videoRef.current) {
      cameraStreamRef.current = stream;
      videoRef.current.srcObject = stream;
      videoRef.current.style.transform = mode === 'user' ? 'scaleX(-1)' : 'none';
      videoRef.current.play().catch(() => {});
      setCameraReady(true);
      setCameraPermission('granted');
      setCameraHint('Camera connected.');
    }
  }

  function requestCameraAgain(facing) {
    const mode = facing ?? facingModeRef.current;
    setError('');
    setCameraHint('Requesting camera access...');
    startCamera(mode).catch(() => {
      setCameraReady(false);
      setCameraPermission('denied');
      setCameraHint('Still blocked. Enable permission in browser settings and refresh if needed.');
    });
  }

  function flipCamera() {
    const next = facingModeRef.current === 'user' ? 'environment' : 'user';
    facingModeRef.current = next;
    requestCameraAgain(next);
  }

  function speakFeedback(text, isGood) {
    if (!voiceEnabledRef.current || !text) return;
    const now = Date.now();
    const cooldown = isGood ? VOICE_COOLDOWN_GOOD_MS : VOICE_COOLDOWN_MS;
    if (text === lastSpokenRef.current.text && now - lastSpokenRef.current.at < cooldown) return;
    lastSpokenRef.current = { text, at: now };

    // Android WebView: use native TTS bridge injected by Flutter
    if (window.NativeTts) {
      window.NativeTts.postMessage(text);
      return;
    }

    // Desktop / other browsers: Web Speech API fallback
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.resume();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.92;
    utterance.pitch = 1;
    utterance.volume = 0.85;
    window.speechSynthesis.speak(utterance);
  }

  // Android WebView periodically pauses speechSynthesis when the window is not
  // in direct focus. Resuming every second keeps the queue alive.
  useEffect(() => {
    if (!voiceEnabled || !('speechSynthesis' in window)) {
      return undefined;
    }
    const id = setInterval(() => {
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }
    }, 1000);
    return () => clearInterval(id);
  }, [voiceEnabled]);

  function handleVoiceToggle(enabled) {
    setVoiceEnabled(enabled);

    // Native TTS (Android WebView) — no warm-up needed
    if (window.NativeTts) {
      if (enabled) lastSpokenRef.current = { text: '', at: 0 };
      return;
    }

    // Web Speech API (desktop)
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    if (enabled) {
      window.speechSynthesis.resume();
      lastSpokenRef.current = { text: '', at: 0 };
      const warmup = new SpeechSynthesisUtterance('Voice on');
      warmup.volume = 0.01;
      window.speechSynthesis.speak(warmup);
    }
  }

  function feedbackKey(result) {
    if (!result?.success) {
      return `camera:${result?.error || 'not_ready'}`;
    }

    return `${result.pose || selectedPoseRef.current}:${result.phase || 'hold'}:${result.issue || result.status}`;
  }

  function acceptStableFeedback(result) {
    if (!result) {
      return false;
    }

    if (!result.success) {
      feedbackWindowRef.current = [];
      setStabilityMessage('');
      return true;
    }

    const key = feedbackKey(result);
    const nextWindow = [...feedbackWindowRef.current, key].slice(-STABILITY_WINDOW);
    feedbackWindowRef.current = nextWindow;

    const matches = nextWindow.filter((item) => item === key).length;
    const isCorrection = result.status !== 'good';
    const requiredMatches = isCorrection ? STABILITY_REQUIRED : 2;
    const isStable = matches >= requiredMatches;

    setStabilityMessage(isStable ? '' : 'Checking consistency...');
    return isStable;
  }

  function updateWorkoutStats(result) {
    const isSquat = selectedPoseRef.current === 'squat';
    setPoseDetected(Boolean(result?.success));

    if (!isSquat) {
      squatTrackerRef.current = { seenBottom: false, lastPhase: '', lastRepAt: 0 };
      return;
    }

    if (!result?.success || result.pose !== 'squat') {
      return;
    }

    const phase = result.phase || '';
    const tracker = squatTrackerRef.current;

    if (phase === 'bottom') {
      tracker.seenBottom = true;
    }

    if (tracker.seenBottom && phase === 'standing' && tracker.lastPhase !== 'standing') {
      const now = Date.now();
      if (now - tracker.lastRepAt > 1800) {
        setSquatReps((count) => count + 1);
        tracker.lastRepAt = now;
      }
      tracker.seenBottom = false;
    }

    tracker.lastPhase = phase;
  }

  function resetSession() {
    setSessionSeconds(0);
    setPoseSeconds(0);
    setSquatReps(0);
    setPoseDetected(false);
    feedbackRef.current = null;
    metricsRef.current = null;
    squatTrackerRef.current = { seenBottom: false, lastPhase: '', lastRepAt: 0 };
  }

  function jumpToCoach(autostart = false) {
    setShowCoach(true);
    requestAnimationFrame(() => {
      coachSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    if (!cameraReady) {
      requestCameraAgain();
    } else if (autostart) {
      setIsRealtime(true);
    }
  }

  function addHistoryEntry(result, latencyMs) {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      time: new Date().toLocaleTimeString(),
      pose: selectedPoseRef.current,
      status: result?.status || (result?.success ? 'good' : 'warning'),
      feedback: result?.feedback || result?.error || 'No feedback',
      issue: result?.issue || '-',
      phase: result?.phase || '-',
      visibility: result?.visibility?.score ?? '-',
      latency: latencyMs,
    };

    setHistory((prev) => [entry, ...prev].slice(0, 25));
  }

  function updateRuntimeStats(result, latencyMs) {
    const next = runtimeStatsRef.current;
    next.count += 1;
    if (result?.success) {
      next.success += 1;
    }
    next.avgLatency = Math.round(((next.avgLatency * (next.count - 1)) + latencyMs) / next.count);
    runtimeStatsRef.current = next;

    setAnalysisCount(next.count);
    setSuccessCount(next.success);
    setAvgLatencyMs(next.avgLatency);
    setLastLatencyMs(latencyMs);
  }

  async function analyzeFrame({ silent = false } = {}) {
    if (!videoRef.current || isAnalyzingRef.current) {
      return;
    }

    isAnalyzingRef.current = true;
    if (!silent) {
      setLoading(true);
      setError('');
    }

    try {
      const startedAt = performance.now();
      const video = videoRef.current;

      if (!poseLandmarkerRef.current) {
        throw new Error('MediaPipe is still loading');
      }

      const results = poseLandmarkerRef.current.detectForVideo(video, startedAt);
      const landmarks = results.landmarks?.[0] || [];

      const payload = {
        selectedPose: selectedPoseRef.current,
        landmarks,
        previousFeedback: feedbackRef.current?.feedback,
        previousMetrics: metricsRef.current,
      };

      const { data } = await axios.post('/api/analyze', payload, {
        headers: { 'Content-Type': 'application/json' }
      });

      const latencyMs = Math.round(performance.now() - startedAt);

      setLastUpdated(new Date().toLocaleTimeString());
      updateWorkoutStats(data);
      updateRuntimeStats(data, latencyMs);
      addHistoryEntry(data, latencyMs);
      speakFeedback(data.feedback, data.status === 'good');
      if (acceptStableFeedback(data)) {
        setFeedback(data);
        feedbackRef.current = data;
        metricsRef.current = data.metrics || null;
      }
    } catch (err) {
      const message = err.response?.data?.detail || err.response?.data?.error || err.message;
      setError(message || 'Analysis failed');
    } finally {
      isAnalyzingRef.current = false;
      if (!silent) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    if (!isRealtime || !cameraReady) {
      return undefined;
    }

    let stopped = false;
    let timerId;

    async function loop() {
      if (stopped) {
        return;
      }
      const startedAt = performance.now();
      await analyzeFrame({ silent: true });
      const elapsed = performance.now() - startedAt;
      if (!stopped) {
        timerId = setTimeout(loop, Math.max(260, intervalMs - elapsed));
      }
    }

    loop();

    return () => {
      stopped = true;
      clearTimeout(timerId);
    };
  }, [isRealtime, intervalMs, cameraReady]);

  useEffect(() => {
    if (!isRealtime) {
      return undefined;
    }

    const timerId = setInterval(() => {
      setSessionSeconds((seconds) => seconds + 1);
      if (poseDetected) {
        setPoseSeconds((seconds) => seconds + 1);
      }
    }, 1000);

    return () => clearInterval(timerId);
  }, [isRealtime, poseDetected]);

  useEffect(() => {
    feedbackWindowRef.current = [];
    setFeedback(null);
    setStabilityMessage('');
    setPoseSeconds(0);
    setPoseDetected(false);
    feedbackRef.current = null;
    metricsRef.current = null;
    squatTrackerRef.current = { seenBottom: false, lastPhase: '', lastRepAt: 0 };
    lastSpokenRef.current = { text: '', at: 0 };
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, [selectedPose]);

  const statusLabel = !cameraReady
    ? 'Camera setup'
    : isRealtime
      ? 'Live analysis running'
      : 'Ready to start';
  const detectRate = analysisCount > 0 ? Math.round((successCount / analysisCount) * 100) : 0;

  /* ── Mobile layout (Flutter WebView) ─────────────────────── */
  if (isMobile) {
    return (
      <div className="m-shell">
        {/* ── Fixed top pane: header + pose row + camera ── */}
        <div className="m-top-pane">
          {/* Header */}
          <div className="m-header">
            <div className="m-brand">
              <span className="brand-mark" />
              <span>GYM BUDDY</span>
            </div>
            <div className="m-header-right">
              <div className="m-live-badge">
                <span className={`dot ${isRealtime ? 'live' : ''}`} />
                {isRealtime ? 'LIVE' : 'PAUSED'}
              </div>
            </div>
          </div>

          {/* Pose selector */}
          <div className="m-pose-row">
            {POSES.map((p) => (
              <button
                key={p.value}
                className={`m-pose-btn ${selectedPose === p.value ? 'active' : ''}`}
                onClick={() => setSelectedPose(p.value)}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Camera */}
          <div className="m-video-wrap">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} className="hidden-canvas" />

            {/* Flip camera button — always visible when camera is ready */}
            {cameraReady && (
              <button className="m-flip-btn" onClick={flipCamera} title="Flip camera">
                ⟳
              </button>
            )}

            {!cameraReady && (
              <div className="m-cam-overlay">
                <p>Tap to enable camera</p>
                <button className="m-btn-primary" onClick={() => requestCameraAgain()}>
                  Enable Camera
                </button>
              </div>
            )}
            {cameraReady && !isRealtime && (
              <div className="m-cam-overlay transparent">
                <button className="m-btn-primary large" onClick={() => setIsRealtime(true)}>
                  ▶ Start Analysis
                </button>
              </div>
            )}
            {isRealtime && feedback && (
              <div className={`m-overlay-badge ${feedback.status === 'good' ? 'good' : 'warn'}`}>
                {feedback.status === 'good' ? '✓ Good Form' : '⚠ Adjust Form'}
              </div>
            )}
          </div>
        </div>

        {/* ── Scrollable bottom pane ── */}
        <div className="m-scroll-pane">
          {/* Controls */}
          <div className="m-controls">
            <button
              className={`m-btn-primary ${isRealtime ? 'danger' : ''}`}
              onClick={() => {
                if (!cameraReady) { requestCameraAgain(); return; }
                setIsRealtime((p) => !p);
              }}
            >
              {isRealtime ? '■ Stop' : '▶ Start Live'}
            </button>
            <button className="m-btn-secondary" onClick={() => analyzeFrame({ silent: false })} disabled={!cameraReady || loading}>
              {loading ? '...' : 'Snap'}
            </button>
            {/* Voice button — must be a real click handler so Android WebView
                unlocks the audio context in a user-gesture callback */}
            <button
              className={`m-btn-secondary ${voiceEnabled ? 'active-voice' : ''}`}
              onClick={() => handleVoiceToggle(!voiceEnabled)}
            >
              {voiceEnabled ? '🔊 Voice' : '🔇 Voice'}
            </button>
            <select className="m-select" value={intervalMs} onChange={(e) => setIntervalMs(Number(e.target.value))}>
              {INTERVAL_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            <button className="m-btn-secondary" onClick={resetSession}>Reset</button>
          </div>

          {/* Tip */}
          {feedback && (
            <div className="m-tip">
              <p className={feedback.status === 'good' ? 'm-status-good' : 'm-status-warn'}>
                {!feedback.success ? 'No pose detected' : feedback.status === 'good' ? 'Good Form' : 'Needs Adjustment'}
              </p>
              <p className="m-feedback-text">{feedback.feedback}</p>
            </div>
          )}

          {error && <p className="m-error">{error}</p>}

          {/* Stats row */}
          <div className="m-stats-row">
            <div className="m-stat"><span>Time</span><strong>{formatDuration(sessionSeconds)}</strong></div>
            <div className="m-stat"><span>{selectedPose === 'squat' ? 'Reps' : 'Hold'}</span><strong>{selectedPose === 'squat' ? squatReps : formatDuration(poseSeconds)}</strong></div>
            <div className="m-stat"><span>Analyses</span><strong>{analysisCount}</strong></div>
            <div className="m-stat"><span>Detection</span><strong>{detectRate}%</strong></div>
            <div className="m-stat"><span>Latency</span><strong>{avgLatencyMs}ms</strong></div>
          </div>

          {/* Metrics */}
          {feedback?.success && (
            <div className="m-metrics">
              {buildMetricRows(feedback, selectedPose, lastUpdated).map(([label, value]) => (
                <div className="m-metric" key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          )}

          {/* History */}
          <div className="m-history">
            <div className="m-history-header">
              <span>History ({history.length})</span>
              <button className="m-btn-ghost" onClick={() => setHistory([])} disabled={history.length === 0}>Clear</button>
            </div>
            {history.length === 0 && <p className="m-muted">No history yet.</p>}
            <ul className="m-history-list">
              {history.map((item) => (
                <li key={item.id} className={item.status === 'good' ? 'good' : 'warn'}>
                  <div>
                    <p className="m-h-main">{item.pose.toUpperCase()} — {item.status}</p>
                    <p className="m-h-sub">{item.feedback}</p>
                  </div>
                  <div className="m-h-meta">
                    <span>{item.time}</span>
                    <span>{item.latency}ms</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  }

  /* ── Desktop / browser layout ─────────────────────────────── */
  return (
    <div className="app-shell">
      <nav className="top-nav">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <p>YOUR GYM BUDDY</p>
            <strong>Smart Pose Coach</strong>
          </div>
        </div>
        <div className="nav-badges">
          <span>{POSES.length} Pose Modes</span>
          <span>Real-Time Feedback</span>
          <span>Voice Cues</span>
          <span>History: {history.length}</span>
        </div>
      </nav>

      <header className="landing-hero">
        <div className="landing-copy">
          <p className="eyebrow">POSTURE AI TRAINER</p>
          <h1>Modern Form Coaching For Every Workout Session</h1>
          <p className="hero-sub">
            A clean training dashboard that reads body landmarks from your camera and gives precise, immediate coaching for strength and yoga movements.
          </p>
          <p className="hero-status">
            <span className={`dot ${isRealtime ? 'live' : ''}`} />
            {statusLabel}
          </p>
          <p className="camera-hint">{cameraHint}</p>
          <div className="landing-actions">
            <button className="btn-primary" onClick={() => jumpToCoach(true)}>
              Start Live Coach
            </button>
            <button className="btn-secondary" onClick={() => jumpToCoach(false)}>
              Explore Dashboard
            </button>
            <button className="btn-ghost" onClick={requestCameraAgain}>
              Retry Camera Permission
            </button>
          </div>
        </div>

        <div className="landing-stats">
          <div className="stat-card">
            <span>Session Time</span>
            <strong>{formatDuration(sessionSeconds)}</strong>
          </div>
          <div className="stat-card">
            <span>{selectedPose === 'squat' ? 'Squat Reps' : 'Pose Hold'}</span>
            <strong>{selectedPose === 'squat' ? squatReps : formatDuration(poseSeconds)}</strong>
          </div>
          <div className="stat-card live-state">
            <span>Live Status</span>
            <strong>{isRealtime ? 'Active' : 'Paused'}</strong>
            <p className={`dot-label ${isRealtime ? 'live' : ''}`}>
              <span className={`dot ${isRealtime ? 'live' : ''}`} />
              {isRealtime ? 'Analyzing in real time' : 'Waiting to start'}
            </p>
          </div>
          <div className="stat-card">
            <span>Camera Permission</span>
            <strong className="permission-badge">{cameraPermission}</strong>
          </div>
        </div>
      </header>

      <section className="feature-strip">
        <article>
          <h3>Pose Recognition</h3>
          <p>Tracks body landmarks and evaluates form quality with actionable cues.</p>
        </article>
        <article>
          <h3>Voice Coaching</h3>
          <p>Optional spoken alerts help you correct posture without looking at the screen.</p>
        </article>
        <article>
          <h3>Performance Stats</h3>
          <p>Session duration, pose time, and squat rep counting for progress tracking.</p>
        </article>
      </section>

      {!showCoach && (
        <section className="panel intro-panel">
          <h2>Ready To Analyze Your Form?</h2>
          <p className="muted">
            Start with a clean landing experience. When you are ready, open the analyzer and begin live posture coaching.
          </p>
          <div className="landing-actions">
            <button className="btn-primary" onClick={() => jumpToCoach(true)}>
              Open Analyzer
            </button>
            <button className="btn-secondary" onClick={() => jumpToCoach(false)}>
              Open Without Auto Start
            </button>
          </div>
        </section>
      )}

      {showCoach && (
      <main className="layout" ref={coachSectionRef}>
        <section className="panel camera-panel">
          <h2>Live Coach Studio</h2>
          <div className="controls-grid">
            <div className="field">
              <label htmlFor="pose">Target Pose</label>
              <select
                id="pose"
                value={selectedPose}
                onChange={(e) => setSelectedPose(e.target.value)}
              >
                {POSES.map((pose) => (
                  <option key={pose.value} value={pose.value}>
                    {pose.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="interval">Update Rate</label>
              <select
                id="interval"
                value={intervalMs}
                onChange={(e) => setIntervalMs(Number(e.target.value))}
              >
                {INTERVAL_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="actions">
            <button
              className="btn-primary"
              onClick={() => setIsRealtime((prev) => !prev)}
              disabled={!cameraReady}
            >
              {isRealtime ? 'Stop Real-Time Coach' : 'Start Real-Time Coach'}
            </button>
            <button className="btn-secondary" onClick={() => analyzeFrame({ silent: false })} disabled={!cameraReady || loading}>
              {loading ? 'Analyzing...' : 'Analyze One Frame'}
            </button>
            <label className="voice-toggle">
              <input
                type="checkbox"
                checked={voiceEnabled}
                onChange={(e) => handleVoiceToggle(e.target.checked)}
              />
              Voice cues
            </label>
            <button className="btn-secondary" onClick={resetSession}>
              Reset Stats
            </button>
          </div>

          <div className="video-wrap">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} className="hidden-canvas" />
            {!cameraReady && <div className="overlay">Waiting for camera...</div>}
          </div>
        </section>

        <section className="panel feedback-panel">
          <h2>Coach Feedback</h2>

          <div className="top-metrics">
            <div className="mini-stat"><span>Analyses</span><strong>{analysisCount}</strong></div>
            <div className="mini-stat"><span>Detection Rate</span><strong>{detectRate}%</strong></div>
            <div className="mini-stat"><span>Avg Latency</span><strong>{avgLatencyMs || 0} ms</strong></div>
            <div className="mini-stat"><span>Last Request</span><strong>{lastLatencyMs || 0} ms</strong></div>
          </div>

          <div className="session-stats">
            <p>
              <span>Workout Time</span>
              <strong>{formatDuration(sessionSeconds)}</strong>
            </p>
            <p>
              <span>{selectedPose === 'squat' ? 'Squat Time' : 'Pose Hold'}</span>
              <strong>{formatDuration(poseSeconds)}</strong>
            </p>
            {selectedPose === 'squat' && (
              <p>
                <span>Squat Reps</span>
                <strong>{squatReps}</strong>
              </p>
            )}
          </div>

          {error && <p className="error">{error}</p>}

          {!error && !feedback && <p className="muted">Start real-time mode or analyze one frame.</p>}
          {!error && feedback && stabilityMessage && <p className="muted">{stabilityMessage}</p>}

          {feedback && (
            <>
              <p className={feedback.status === 'good' ? 'status-good' : 'status-warn'}>
                {!feedback.success
                  ? 'Adjust Camera'
                  : feedback.status === 'good'
                    ? 'Good Form'
                    : 'Needs Adjustment'}
              </p>
              <p className="tip">{feedback.feedback}</p>

              <div className="metrics">
                {buildMetricRows(feedback, selectedPose, lastUpdated).map(([label, value]) => (
                  <p key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </p>
                ))}
              </div>
            </>
          )}

          <div className="history-block">
            <div className="history-header">
              <h3>Recent Analysis History</h3>
              <button className="btn-ghost" onClick={() => setHistory([])} disabled={history.length === 0}>
                Clear
              </button>
            </div>

            {history.length === 0 && <p className="muted">No history yet. Run analysis to populate this section.</p>}

            {history.length > 0 && (
              <ul className="history-list">
                {history.map((item) => (
                  <li key={item.id}>
                    <div>
                      <p className="history-main">{item.pose.toUpperCase()} - {item.status}</p>
                      <p className="history-sub">{item.feedback}</p>
                    </div>
                    <div className="history-meta">
                      <span>{item.time}</span>
                      <span>{item.latency} ms</span>
                      <span>V:{item.visibility}</span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </main>
      )}
    </div>
  );
}
