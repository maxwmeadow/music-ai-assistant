"use client";

import { useState } from "react";
import { CodeEditor } from "@/components/CodeEditor";
import { MixerPanel } from "@/components/MixerPanel";
import { api } from "@/lib/api";
import { RecorderControls } from "@/components/RecorderControls";
import { parseTracksFromDSL, ParsedTrack } from "@/lib/dslParser";
import { Mic, Music, Drum, Play, Square, Sparkles, Sliders, Piano, ChevronDown, ChevronUp } from "lucide-react";
import { Timeline } from "@/components/Timeline/Timeline";
import { usePlaybackTime } from "@/hooks/usePlaybackTime";
import { PianoRoll } from "@/components/PianoRoll/PianoRoll";
import DetectionTuner from "@/components/DetectionTuner";
import { compileIR, VisualizationData } from "@/lib/hum2melody-api";

export default function Home() {
  const [code, setCode] = useState("// Your generated music code will appear here...");
  const [loadingTest, setLoadingTest] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingPlay, setLoadingPlay] = useState(false);
  const [executableCode, setExecutableCode] = useState<string>("");
  const [toast, setToast] = useState<string | null>(null);
  const [tracks, setTracks] = useState<ParsedTrack[]>([]);
  const [trackVolumes, setTrackVolumes] = useState<Record<string, number>>({});
  const [isPlaying, setIsPlaying] = useState(false);
  const currentTime = usePlaybackTime(isPlaying);
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

  const showToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 3000);
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
      setExecutableCode(data.meta?.executable_code || "");

      const parsedTracks = parseTracksFromDSL(code);
      setTracks(parsedTracks);

      showToast("Code compiled successfully");
    } catch (error) {
      console.error(error);
      showToast("Compilation failed");
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

    setLoadingPlay(true);
    setIsPlaying(true);
    try {
      const Tone = await import('tone');
      (window as any).Tone = Tone;
      (window as any).__trackVolumes = trackVolumes;

      eval(executableCode);

      setTimeout(() => {
        Object.entries(trackVolumes).forEach(([trackId, volume]) => {
          handleVolumeChange(trackId, volume);
        });
      }, 100);

      const tempoMatch = code.match(/tempo\((\d+)\)/);
      const tempo = tempoMatch ? parseInt(tempoMatch[1]) : 120;

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

        const chordRegex = /chord\(\[[^\]]+\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
        let chordMatch;

        while ((chordMatch = chordRegex.exec(trackContent)) !== null) {
          const start = parseFloat(chordMatch[1]);
          const duration = parseFloat(chordMatch[2]);
          const endTime = start + duration;
          maxDuration = Math.max(maxDuration, endTime);
        }
      }

      console.log(`[Playback] Calculated duration: ${maxDuration.toFixed(2)}s`);

      const stopTimeout = setTimeout(() => {
        stopAudio();
      }, (maxDuration + 1.5) * 1000);

      (window as any).__autoStopTimeout = stopTimeout;

      showToast(`Playing (${maxDuration.toFixed(1)}s)...`);
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
    }

    if ((window as any).__musicControls?.stop) {
      (window as any).__musicControls.stop();
      setIsPlaying(false);
      showToast("Stopped");
    }
  };

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

  return (
    <div className="h-screen flex flex-col bg-[#1a1a1a] overflow-hidden">
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
              disabled={loadingPlay || !executableCode}
              onClick={playAudio}
              className="flex items-center justify-center w-10 h-10 bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
              title="Play (Space)"
            >
              <Play className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={stopAudio}
              className="flex items-center justify-center w-10 h-10 bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors"
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
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT PANEL - Timeline */}
        <div className="w-1/2 flex flex-col bg-[#1e1e1e] border-r border-gray-800 min-w-0">
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

        {/* RIGHT PANEL - Code Editor */}
        <div className="w-1/2 flex flex-col bg-[#1e1e1e] min-w-0">
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
