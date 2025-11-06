"use client";

import { useState, useEffect } from "react";
import { CodeEditor } from "@/components/CodeEditor";
import { MixerPanel } from "@/components/MixerPanel";
import { api } from "@/lib/api";
import { RecorderControls } from "@/components/RecorderControls";
import { parseTracksFromDSL, ParsedTrack } from "@/lib/dslParser";
import { Mic, Music, Drum, Play, Square, Sparkles, Sliders, Piano, ChevronDown, Undo, Redo } from "lucide-react";
import { Timeline } from "@/components/Timeline/Timeline";
import { PianoRoll } from "@/components/PianoRoll/PianoRoll";
import DetectionTuner from "@/components/DetectionTuner";
import { compileIR, VisualizationData } from "@/lib/hum2melody-api";
import { useHistory } from "@/hooks/useHistory";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";

export default function Home() {
  const { pushHistory, undo, redo, canUndo, canRedo, currentState: code } = useHistory("// Your generated music code will appear here...");

  // Helper to update code and push to history
  const setCode = (newCode: string) => {
    pushHistory(newCode);
  };

  // Keyboard shortcuts modal state
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false);

  const [loadingTest, setLoadingTest] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingPlay, setLoadingPlay] = useState(false);
  const [executableCode, setExecutableCode] = useState<string>("");
  const [toast, setToast] = useState<string | null>(null);
  const [tracks, setTracks] = useState<ParsedTrack[]>([]);
  const [trackVolumes, setTrackVolumes] = useState<Record<string, number>>({});
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [selectedTrackForPianoRoll, setSelectedTrackForPianoRoll] = useState<string | null>(null);

  // Detection Tuning State
  const [tuningMode, setTuningMode] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [currentIR, setCurrentIR] = useState<any>(null);

  // Panel visibility
  const [showMixer, setShowMixer] = useState(false);
  const [showRecorder, setShowRecorder] = useState(false);
  const [recordingMode, setRecordingMode] = useState<'melody' | 'drums' | null>(null);

  // Resizable panels
  const [leftPanelWidth, setLeftPanelWidth] = useState(50); // percentage
  const [isResizing, setIsResizing] = useState(false);

  const showToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 3000);
  };

  // Calculate the maximum duration of the current music from DSL code
  const calculateMaxDuration = (): number => {
    let maxDuration = 0;
    const trackRegex = /track\("([^"]+)"\)\s*\{([^}]+)\}/g;
    let trackMatch;

    while ((trackMatch = trackRegex.exec(code)) !== null) {
      const trackContent = trackMatch[2];
      const noteRegex = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
      let noteMatch;

      while ((noteMatch = noteRegex.exec(trackContent)) !== null) {
        const start = parseFloat(noteMatch[2]);
        const duration = parseFloat(noteMatch[3]);
        const endTime = start + duration;
        maxDuration = Math.max(maxDuration, endTime);
      }

      const chordRegex = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
      let chordMatch;

      while ((chordMatch = chordRegex.exec(trackContent)) !== null) {
        const start = parseFloat(chordMatch[1]);
        const duration = parseFloat(chordMatch[2]);
        const endTime = start + duration;
        maxDuration = Math.max(maxDuration, endTime);
      }
    }

    return maxDuration;
  };

  const fetchTest = async () => {
    setLoadingTest(true);
    try {
      const response = await api("/test");
      const dslCode = await response.text();
      setCode(dslCode);

      const parsedTracks = parseTracksFromDSL(dslCode);
      setTracks(parsedTracks);

      showToast("Sample code loaded");
    } catch (error) {
      console.error(error);
      showToast("Failed to load sample");
    } finally {
      setLoadingTest(false);
    }
  };

  const sendToRunner = async () => {
    setLoadingRun(true);
    try {
      const response = await api("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });

      const data = await response.json();
      const execCode = data.meta?.executable_code || "";
      setExecutableCode(execCode);

      const parsedTracks = parseTracksFromDSL(code);
      setTracks(parsedTracks);

      // Preload audio samples by evaluating the code without starting playback
      if (execCode) {
        setIsLoadingAudio(true);
        (window as any).__isPreloading = true; // Flag to prevent console interceptor from clearing loading
        try {
          const Tone = await import('tone');
          (window as any).Tone = Tone;
          (window as any).__trackVolumes = trackVolumes;

          // Stop any existing playback before preloading
          if (Tone.Transport.state === 'started') {
            Tone.Transport.stop();
          }
          Tone.Transport.seconds = 0;
          setCurrentTime(0);
          setIsPlaying(false);

          // Temporarily replace Transport.start() to prevent autoplay during preload
          const originalStart = Tone.Transport.start;
          let startWasCalled = false;
          Tone.Transport.start = function(...args: any[]) {
            console.log('[Compile] Intercepted Transport.start() - blocking during preload');
            startWasCalled = true;
            return Tone.Transport as any; // Return this for chaining
          };

          try {
            // Evaluate the code to trigger sample loading
            // Transport.start() is now a no-op, so no playback occurs
            eval(execCode);
          } finally {
            // Restore original Transport.start()
            Tone.Transport.start = originalStart;

            // Force stop transport after restoring start()
            // This ensures no scheduled events trigger playback
            if (Tone.Transport.state !== 'stopped') {
              Tone.Transport.stop();
            }
            Tone.Transport.seconds = 0;
            Tone.Transport.cancel(); // Cancel all scheduled events
          }

          console.log(`[Compile] Sample preload complete. Transport.start was ${startWasCalled ? 'called and blocked' : 'not called'}`);

          // Mark samples as preloaded
          (window as any).__samplesPreloaded = true;

          // Now trigger actual sample loading by starting the Transport briefly
          // This will cause "[Music] Playing..." to be logged when samples are ready
          console.log('[Compile] Initiating sample loading...');
          Tone.Transport.start();

          // The loading state will be cleared by the console.log interceptor
          // when it detects "[Music] Playing..." message
          // The __isPreloading flag will be cleared at that time too
        } catch (preloadError) {
          console.error('[Compile] Preload error:', preloadError);
          (window as any).__isPreloading = false;
          setIsLoadingAudio(false);
        }
      }

      showToast("Code compiled successfully");
    } catch (error) {
      console.error(error);
      showToast("Compilation failed");
      setIsLoadingAudio(false);
    } finally {
      setLoadingRun(false);
    }
  };

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTrackVolumes(prev => ({ ...prev, [trackId]: volume }));

    if ((window as any).__musicControls?.pools) {
      const pool = (window as any).__musicControls.pools.get(trackId);
      if (pool?.voices) {
        pool.voices.forEach((voice: any) => {
          if (voice.volume) {
            voice.volume.value = volume;
          }
        });
      }
    }
  };

  const playAudio = async () => {
    if (!executableCode) {
      showToast("Please compile the code first");
      return;
    }

    // Don't allow play while audio is still loading
    if (isLoadingAudio) {
      showToast("Audio is still loading...");
      return;
    }

    // Clear any existing auto-stop timeout before starting
    if ((window as any).__autoStopTimeout) {
      clearTimeout((window as any).__autoStopTimeout);
      (window as any).__autoStopTimeout = null;
    }

    setLoadingPlay(true);
    setIsPlaying(true);
    try {
      const Tone = await import('tone');
      (window as any).Tone = Tone;
      (window as any).__trackVolumes = trackVolumes;

      // Check if we're resuming from pause or starting fresh
      const currentPosition = Tone.Transport.seconds;
      const isResuming = currentPosition > 0 && Tone.Transport.state === 'paused';

      // Check if samples are already loaded (preloaded during compile)
      const samplesPreloaded = (window as any).__samplesPreloaded;

      if (!isResuming && !samplesPreloaded) {
        // Set loading state before eval (only if not preloaded)
        setIsLoadingAudio(true);

        // Only eval code if starting fresh and not preloaded
        eval(executableCode);

        // Mark as preloaded
        (window as any).__samplesPreloaded = true;

        // Loading will be cleared when "Playing..." is logged
      }

      // Start or resume transport
      Tone.Transport.start();

      setTimeout(() => {
        Object.entries(trackVolumes).forEach(([trackId, volume]) => {
          handleVolumeChange(trackId, volume);
        });
      }, 100);

      const maxDuration = calculateMaxDuration();
      console.log(`[Playback] Calculated duration: ${maxDuration.toFixed(2)}s`);

      // Calculate remaining time from current position
      const remainingTime = Math.max(0, maxDuration - currentPosition);

      const stopTimeout = setTimeout(() => {
        // Only auto-stop if still playing (not manually paused)
        if (Tone.Transport.state === 'started') {
          stopAudio();
        }
      }, (remainingTime + 1.5) * 1000);

      (window as any).__autoStopTimeout = stopTimeout;

      showToast(isResuming ? 'Resumed' : `Playing (${maxDuration.toFixed(1)}s)...`);
    } catch (error) {
      console.error(error);
      showToast("Playback error");
      setIsPlaying(false);
    } finally {
      setLoadingPlay(false);
    }
  };

  const stopAudio = () => {
    if ((window as any).__autoStopTimeout) {
      clearTimeout((window as any).__autoStopTimeout);
      (window as any).__autoStopTimeout = null;
    }

    // Use pause instead of stop to preserve playhead position
    if ((window as any).Tone?.Transport) {
      const wasPlaying = (window as any).Tone.Transport.state === 'started';
      (window as any).Tone.Transport.pause();
      setIsPlaying(false);
      if (wasPlaying) {
        showToast("Paused");
      }
    }
  };

  const resetAudio = () => {
    if ((window as any).__autoStopTimeout) {
      clearTimeout((window as any).__autoStopTimeout);
      (window as any).__autoStopTimeout = null;
    }

    if ((window as any).Tone?.Transport) {
      (window as any).Tone.Transport.stop();
      (window as any).Tone.Transport.seconds = 0;
      setCurrentTime(0);
      setIsPlaying(false);
      showToast("Stopped");
    }
  };

  const handleSeek = (time: number) => {
    // Don't allow seeking while audio is loading
    if (isLoadingAudio) {
      return;
    }

    if ((window as any).Tone?.Transport) {
      (window as any).Tone.Transport.seconds = time;
      setCurrentTime(time);

      // If playing, recalculate the auto-stop timeout
      if (isPlaying && (window as any).Tone.Transport.state === 'started') {
        // Clear existing timeout
        if ((window as any).__autoStopTimeout) {
          clearTimeout((window as any).__autoStopTimeout);
        }

        // Calculate remaining time from new position
        const maxDuration = calculateMaxDuration();
        const remainingTime = Math.max(0, maxDuration - time);

        console.log(`[Seek] Recalculating timeout: ${remainingTime.toFixed(2)}s remaining from position ${time.toFixed(2)}s`);

        // Set new timeout
        const stopTimeout = setTimeout(() => {
          if ((window as any).Tone?.Transport.state === 'started') {
            stopAudio();
          }
        }, (remainingTime + 1.5) * 1000);

        (window as any).__autoStopTimeout = stopTimeout;
      }
    }
  };

  // Poll Tone.js transport time continuously
  useEffect(() => {
    const interval = setInterval(() => {
      if ((window as any).Tone?.Transport) {
        const transportTime = (window as any).Tone.Transport.seconds;
        setCurrentTime(transportTime);
      }
      // Don't reset to 0 - let the last known position persist
    }, isPlaying ? 50 : 100); // Poll faster when playing

    return () => clearInterval(interval);
  }, [isPlaying]);

  // Listen for audio loading events
  useEffect(() => {
    const handleAudioReady = () => {
      console.log('[Timeline] Audio ready event received');
      setIsLoadingAudio(false);
    };

    const handlePlaybackComplete = () => {
      console.log('[Timeline] Playback complete event received');
      setIsLoadingAudio(false);
    };

    window.addEventListener('audioReady', handleAudioReady);
    window.addEventListener('playbackComplete', handlePlaybackComplete);

    return () => {
      window.removeEventListener('audioReady', handleAudioReady);
      window.removeEventListener('playbackComplete', handlePlaybackComplete);
    };
  }, []);

  // Also intercept console.log as fallback to detect audio loading events
  useEffect(() => {
    const originalLog = console.log;

    console.log = function(...args: any[]) {
      // Call original console.log
      originalLog.apply(console, args);

      // Check for specific messages
      const message = args.join(' ');

      // Check if we're in preload mode
      const isPreloading = (window as any).__isPreloading;

      if (message.includes('[Music] Playing...')) {
        if (isPreloading) {
          // We're in preload mode - stop the Transport immediately
          console.log('[Compile] Samples loaded, stopping preload playback');
          if ((window as any).Tone?.Transport) {
            (window as any).Tone.Transport.stop();
            (window as any).Tone.Transport.seconds = 0;
          }
          // Clear preloading flag and loading state
          (window as any).__isPreloading = false;
          setIsLoadingAudio(false);
        } else {
          // Normal playback mode
          setIsLoadingAudio(false);
        }
      } else if (message.includes('[Music] Playback complete') && !isPreloading) {
        setIsLoadingAudio(false);
      }
    };

    return () => {
      console.log = originalLog;
    };
  }, []);

  const handleMelodyGenerated = async (result: any) => {
    console.log("[DEBUG] handleMelodyGenerated called with:", result);
    console.log("[DEBUG] Has visualization?", !!result.visualization);
    console.log("[DEBUG] Has session_id?", !!result.session_id);

    try {
      if (result.visualization && result.session_id) {
        console.log("[DEBUG] Opening tuning modal with session:", result.session_id);
        setSessionId(result.session_id);
        setVisualizationData(result.visualization);
        setCurrentIR(result.ir);
        setTuningMode(true);
        showToast("Tuning interface ready - adjust parameters to improve detection");
      } else {
        console.log("[DEBUG] No visualization data, using IR directly");
        await applyIRAndCompile(result.ir);
      }
    } catch (error) {
      console.error("Failed to process melody:", error);
      showToast("Failed to process melody");
    }

    // Close recorder after generation
    setShowRecorder(false);
    setRecordingMode(null);
  };

  const applyIRAndCompile = async (ir: any) => {
    showToast("Converting to DSL...");

    try {
      const response = await api("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ir }),
      });

      const data = await response.json();

      if (data.dsl) {
        setCode(data.dsl);
        setExecutableCode(data.meta?.executable_code || "");

        const parsedTracks = parseTracksFromDSL(data.dsl);
        setTracks(parsedTracks);

        showToast("Melody loaded! Click compile & play");
      } else {
        showToast("Failed to convert to DSL");
      }
    } catch (error) {
      console.error("Failed to convert IR to DSL:", error);
      showToast("Conversion failed");
    }
  };

  const handleApplyTuning = async (finalIR: any) => {
    setTuningMode(false);
    await applyIRAndCompile(finalIR);
  };

  const openRecorder = (mode: 'melody' | 'drums') => {
    setRecordingMode(mode);
    setShowRecorder(true);
  };

  // Handle resize drag
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  // Add/remove global mouse event listeners for resizing
  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = (e.clientX / window.innerWidth) * 100;
      // Constrain between 20% and 80%
      if (newWidth >= 20 && newWidth <= 80) {
        setLeftPanelWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Keyboard shortcuts for undo/redo, play/pause, and help
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if we're in an input/textarea (don't trigger shortcuts while typing)
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;

      // Play/Pause: Space
      if (e.key === ' ' && !ctrlOrCmd) {
        e.preventDefault();
        if (isPlaying) {
          stopAudio();
        } else {
          playAudio();
        }
      }

      // Show keyboard shortcuts: ?
      if (e.key === '?' && !ctrlOrCmd) {
        e.preventDefault();
        setShowKeyboardShortcuts(true);
      }

      // Undo: Ctrl+Z / Cmd+Z
      if (ctrlOrCmd && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
        showToast('Undo');
      }

      // Redo: Ctrl+Y / Cmd+Y or Ctrl+Shift+Z / Cmd+Shift+Z
      if ((ctrlOrCmd && e.key === 'y') || (ctrlOrCmd && e.shiftKey && e.key === 'z')) {
        e.preventDefault();
        redo();
        showToast('Redo');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, isPlaying, isLoadingAudio, executableCode, playAudio, stopAudio]);

  return (
    <div className="h-screen flex flex-col bg-[#1a1a1a] overflow-hidden">
      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcuts
        isOpen={showKeyboardShortcuts}
        onClose={() => setShowKeyboardShortcuts(false)}
      />

      {/* Detection Tuner Modal */}
      {tuningMode && visualizationData && sessionId && currentIR && (
        <DetectionTuner
          sessionId={sessionId}
          initialVisualization={visualizationData}
          initialIR={currentIR}
          onApply={handleApplyTuning}
          onCancel={() => {
            setTuningMode(false);
            showToast("Tuning cancelled - using initial detection");
            applyIRAndCompile(currentIR);
          }}
        />
      )}

      {/* Toast */}
      {toast && (
        <div className="fixed top-4 right-4 bg-[#2a2a2a] border border-gray-700 text-white px-4 py-3 rounded-lg shadow-xl z-50 animate-fade-in">
          {toast}
        </div>
      )}

      {/* Recorder Modal */}
      {showRecorder && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[#252525] border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                {recordingMode === 'melody' ? (
                  <>
                    <Mic className="w-5 h-5 text-blue-400" />
                    Record Melody
                  </>
                ) : (
                  <>
                    <Drum className="w-5 h-5 text-orange-400" />
                    Record Drums
                  </>
                )}
              </h2>
              <button
                onClick={() => {
                  setShowRecorder(false);
                  setRecordingMode(null);
                }}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <RecorderControls onMelodyGenerated={handleMelodyGenerated} />
          </div>
        </div>
      )}

      {/* Top Toolbar */}
      <div className="flex-none h-16 bg-[#252525] border-b border-gray-800 flex items-center px-4 gap-4">
        {/* Logo */}
        <div className="flex items-center gap-2 mr-4">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
            <Music className="w-5 h-5 text-white" />
          </div>
          <span className="text-white font-bold text-lg">Studio</span>
        </div>

        {/* Model Buttons */}
        <div className="flex items-center gap-2 border-r border-gray-700 pr-4">
          <button
            onClick={() => openRecorder('melody')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors text-sm font-medium"
          >
            <Mic className="w-4 h-4" />
            Hum2Melody
          </button>
          <button
            onClick={() => openRecorder('drums')}
            className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg transition-colors text-sm font-medium"
          >
            <Drum className="w-4 h-4" />
            Beatbox2Drums
          </button>
          <button
            disabled
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-gray-400 rounded-lg text-sm font-medium cursor-not-allowed"
            title="Coming soon"
          >
            <Sparkles className="w-4 h-4" />
            Arranger
          </button>
        </div>

        {/* Transport Controls */}
        <div className="flex items-center gap-2 border-r border-gray-700 pr-4">
          {!isPlaying ? (
            <button
              disabled={loadingPlay || !executableCode || isLoadingAudio}
              onClick={playAudio}
              className="flex items-center justify-center w-10 h-10 bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
              title={isLoadingAudio ? "Loading audio..." : "Play (Space)"}
            >
              <Play className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={stopAudio}
              disabled={isLoadingAudio}
              className="flex items-center justify-center w-10 h-10 bg-red-600 hover:bg-red-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
              title="Stop (Space)"
            >
              <Square className="w-5 h-5" />
            </button>
          )}

          <button
            disabled={loadingRun}
            onClick={sendToRunner}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors border border-gray-700"
          >
            {loadingRun ? "Compiling..." : "Compile"}
          </button>
        </div>

        {/* Undo/Redo Controls */}
        <div className="flex items-center gap-2 border-r border-gray-700 pr-4">
          <button
            disabled={!canUndo}
            onClick={undo}
            className="flex items-center justify-center w-9 h-9 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-lg transition-colors border border-gray-700"
            title="Undo (Ctrl+Z)"
          >
            <Undo className="w-4 h-4" />
          </button>
          <button
            disabled={!canRedo}
            onClick={redo}
            className="flex items-center justify-center w-9 h-9 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-lg transition-colors border border-gray-700"
            title="Redo (Ctrl+Y)"
          >
            <Redo className="w-4 h-4" />
          </button>
        </div>

        {/* Panel Toggles */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowMixer(!showMixer)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              showMixer
                ? 'bg-blue-600 text-white'
                : 'bg-[#2a2a2a] text-gray-300 hover:bg-[#333] border border-gray-700'
            }`}
          >
            <Sliders className="w-4 h-4" />
            Mixer
          </button>

          {selectedTrackForPianoRoll ? (
            <button
              onClick={() => setSelectedTrackForPianoRoll(null)}
              className="flex items-center gap-2 px-3 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Piano className="w-4 h-4" />
              Close Piano Roll
            </button>
          ) : tracks.length > 0 && (
            <div className="relative group">
              <button
                className="flex items-center gap-2 px-3 py-2 bg-[#2a2a2a] text-gray-300 hover:bg-[#333] border border-gray-700 rounded-lg text-sm font-medium transition-colors"
              >
                <Piano className="w-4 h-4" />
                Piano Roll
                <ChevronDown className="w-4 h-4" />
              </button>
              <div className="absolute top-full left-0 pt-1 hidden group-hover:block z-50">
                <div className="bg-[#2a2a2a] border border-gray-700 rounded-lg shadow-xl min-w-[150px]">
                  {tracks.map(track => (
                    <button
                      key={track.id}
                      onClick={() => setSelectedTrackForPianoRoll(track.id)}
                      className="block w-full text-left px-4 py-2 text-gray-300 hover:bg-[#333] first:rounded-t-lg last:rounded-b-lg transition-colors"
                    >
                      {track.id}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right side - test button */}
        <div className="ml-auto">
          <button
            disabled={loadingTest}
            onClick={fetchTest}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 text-gray-300 border border-gray-700 rounded-lg text-sm transition-colors"
          >
            {loadingTest ? "Loading..." : "Load Sample"}
          </button>
        </div>
      </div>

      {/* Main Content Area - 2 Panel Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* LEFT PANEL - Timeline */}
        <div
          className="flex flex-col bg-[#1e1e1e] min-w-0"
          style={{ width: `${leftPanelWidth}%` }}
        >
          <div className="flex-none px-4 py-3 bg-[#252525] border-b border-gray-800">
            <h2 className="text-sm font-semibold text-gray-300">ARRANGEMENT</h2>
          </div>
          <div className="flex-1 overflow-hidden p-4">
            {tracks.length > 0 ? (
              <Timeline
                tracks={tracks}
                dslCode={code}
                onCodeChange={setCode}
                isPlaying={isPlaying}
                currentTime={currentTime}
                onSeek={handleSeek}
                isLoading={isLoadingAudio}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <Music className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No tracks yet</p>
                  <p className="text-xs mt-1">Record audio or load sample to get started</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RESIZE BAR */}
        <div
          className={`w-1 bg-gray-800 hover:bg-blue-500 cursor-col-resize flex-shrink-0 relative group transition-colors ${
            isResizing ? 'bg-blue-500' : ''
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Visual indicator on hover */}
          <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-blue-500/20" />
        </div>

        {/* RIGHT PANEL - Code Editor */}
        <div
          className="flex flex-col bg-[#1e1e1e] min-w-0"
          style={{ width: `${100 - leftPanelWidth}%` }}
        >
          <div className="flex-none px-4 py-3 bg-[#252525] border-b border-gray-800">
            <h2 className="text-sm font-semibold text-gray-300">CODE EDITOR</h2>
          </div>
          <div className="flex-1 overflow-hidden">
            <CodeEditor value={code} onChange={setCode} />
          </div>
        </div>
      </div>

      {/* BOTTOM PANEL - Mixer (Toggleable) */}
      {showMixer && tracks.length > 0 && (
        <div className="flex-none h-56 bg-[#252525] border-t border-gray-800 overflow-x-auto overflow-y-hidden p-3">
          <MixerPanel tracks={tracks} onVolumeChange={handleVolumeChange} />
        </div>
      )}

      {/* Piano Roll Overlay */}
      {selectedTrackForPianoRoll && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-40">
          <div className="bg-[#1e1e1e] border border-gray-700 rounded-xl w-[90vw] h-[80vh] flex flex-col shadow-2xl">
            <div className="flex-none px-4 py-3 bg-[#252525] border-b border-gray-800 flex items-center justify-between rounded-t-xl">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Piano className="w-5 h-5 text-purple-400" />
                Piano Roll - {selectedTrackForPianoRoll}
              </h2>
              <button
                onClick={() => setSelectedTrackForPianoRoll(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-hidden p-4">
              <PianoRoll
                track={tracks.find(t => t.id === selectedTrackForPianoRoll)!}
                dslCode={code}
                onCodeChange={setCode}
                isPlaying={isPlaying}
                currentTime={currentTime}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}