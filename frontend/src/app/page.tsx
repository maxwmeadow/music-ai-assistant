"use client";

import { useState } from "react";
import { CodeEditor } from "@/components/CodeEditor";
import { MixerPanel } from "@/components/MixerPanel";
import { api } from "@/lib/api";
import { RecorderControls } from "@/components/RecorderControls";
import { parseTracksFromDSL, ParsedTrack } from "@/lib/dslParser";
import { Mic, Music, Drum, Play, Square, Sparkles } from "lucide-react";
import { Timeline } from "@/components/Timeline/Timeline";
import { usePlaybackTime } from "@/hooks/usePlaybackTime";

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

      // Calculate total duration from DSL
      const tempoMatch = code.match(/tempo\((\d+)\)/);
      const tempo = tempoMatch ? parseInt(tempoMatch[1]) : 120;

      let maxDuration = 0;
      tracks.forEach(track => {
        const trackMatch = code.match(new RegExp(`track\\("${track.id}"\\)\\s*{([^}]+)}`, 's'));
        if (trackMatch) {
          const noteMatches = trackMatch[1].matchAll(/(?:note|chord)\([^)]+,\s*([\d.]+),/g);
          let trackDuration = 0;
          for (const match of noteMatches) {
            trackDuration += parseFloat(match[1]);
          }
          maxDuration = Math.max(maxDuration, trackDuration);
        }
      });

      // Auto-stop when done (add 1 second buffer)
      const stopTimeout = setTimeout(() => {
        stopAudio();
      }, (maxDuration + 1) * 1000);

      // Store timeout to clear if manually stopped
      (window as any).__autoStopTimeout = stopTimeout;

      showToast("Playing...");
    } catch (error) {
      console.error(error);
      showToast("Playback error");
      setIsPlaying(false);
    } finally {
      setLoadingPlay(false);
    }
  };

  const stopAudio = () => {
    // Clear auto-stop timeout
    if ((window as any).__autoStopTimeout) {
      clearTimeout((window as any).__autoStopTimeout);
    }

    if ((window as any).__musicControls?.stop) {
      (window as any).__musicControls.stop();
      setIsPlaying(false);
      showToast("Stopped");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Toast */}
      {toast && (
        <div className="fixed top-6 right-6 bg-white/10 backdrop-blur-lg border border-white/20 text-white px-6 py-3 rounded-xl shadow-2xl z-50 animate-fade-in">
          {toast}
        </div>
      )}

      {/* Header */}
      <div className="border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Music AI Studio</h1>
              <p className="text-sm text-purple-200">Turn your voice into music</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* LEFT: Input Panel */}
          <div className="space-y-6">
            {/* AI Actions */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
              <h2 className="text-xl font-bold text-white mb-4">Create with AI</h2>

              <div className="space-y-3">
                <button className="w-full group relative overflow-hidden bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl px-6 py-4 transition-all duration-300 transform hover:scale-[1.02]">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
                      <Mic className="w-6 h-6" />
                    </div>
                    <div className="text-left flex-1">
                      <p className="font-semibold text-lg">Hum a Melody</p>
                      <p className="text-sm text-purple-100">Transform your humming into musical notes</p>
                    </div>
                  </div>
                </button>

                <button className="w-full group relative overflow-hidden bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white rounded-xl px-6 py-4 transition-all duration-300 transform hover:scale-[1.02]">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
                      <Drum className="w-6 h-6" />
                    </div>
                    <div className="text-left flex-1">
                      <p className="font-semibold text-lg">Beatbox Drums</p>
                      <p className="text-sm text-blue-100">Create drum patterns from beatboxing</p>
                    </div>
                  </div>
                </button>

                <button className="w-full group relative overflow-hidden bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white rounded-xl px-6 py-4 transition-all duration-300 transform hover:scale-[1.02]">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
                      <Music className="w-6 h-6" />
                    </div>
                    <div className="text-left flex-1">
                      <p className="font-semibold text-lg">Add Arrangement</p>
                      <p className="text-sm text-green-100">Generate bass, chords, and more</p>
                    </div>
                  </div>
                </button>
              </div>
            </div>

            {/* Recording */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
              <h2 className="text-xl font-bold text-white mb-4">Record Audio</h2>
              <RecorderControls />
            </div>

            {/* Quick Test */}
            <button
              disabled={loadingTest}
              onClick={fetchTest}
              className="w-full bg-white/10 hover:bg-white/20 border border-white/20 text-white rounded-xl px-6 py-3 transition-all duration-300 disabled:opacity-50 font-medium"
            >
              {loadingTest ? "Loading..." : "Load Sample Code"}
            </button>
          </div>

          {/* RIGHT: Code & Mixer */}
          <div className="space-y-6">
            {/* Code Editor */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white">Music Code</h2>
                <div className="flex gap-2">
                  <button
                    disabled={loadingRun}
                    onClick={sendToRunner}
                    className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
                  >
                    {loadingRun ? "Compiling..." : "Compile"}
                  </button>

                  {!isPlaying ? (
                    <button
                      disabled={loadingPlay || !executableCode}
                      onClick={playAudio}
                      className="bg-green-600 hover:bg-green-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                      <Play className="w-4 h-4" />
                      Play
                    </button>
                  ) : (
                    <button
                      onClick={stopAudio}
                      className="bg-red-600 hover:bg-red-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </button>
                  )}
                </div>
              </div>

              <div className="rounded-xl overflow-hidden border border-white/10">
                <CodeEditor value={code} onChange={setCode} />
              </div>
            </div>

            {/* Mixer */}
            {tracks.length > 0 && (
              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
                <MixerPanel tracks={tracks} onVolumeChange={handleVolumeChange} />
              </div>
            )}
          </div>
        </div>

        {/* Timeline - Full Width */}
        {tracks.length > 0 && (
          <div className="mt-6 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
            <Timeline
              tracks={tracks}
              dslCode={code}
              onCodeChange={setCode}
              isPlaying={isPlaying}
              currentTime={currentTime}
            />
          </div>
        )}
      </div>
    </div>
  );
}