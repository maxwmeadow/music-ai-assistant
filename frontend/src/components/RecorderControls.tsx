// src/components/RecorderControls.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { AudioRecorder } from "@/lib/audioRecorder";
import { WaveformCanvas } from "./WaveformCanvas";
import { processAudioBlob } from "@/lib/audioProcessing";
import { api } from "@/lib/api";
import DrumOnsetVisualizer from "./DrumOnsetVisualizer";
import { Clock, Mic, Play, Square, Sparkles } from "lucide-react";

interface RecorderControlsProps {
    onMelodyGenerated?: (ir: any) => void;
    mode?: 'melody' | 'drums';
    onVisualizationClose?: () => void;
    maxDuration?: number; // Max duration of the current song in seconds
}

export function RecorderControls({
    onMelodyGenerated,
    mode = 'melody',
    onVisualizationClose,
    maxDuration = 0
}: RecorderControlsProps = {
    onMelodyGenerated: undefined,
    mode: 'melody',
    onVisualizationClose: undefined,
    maxDuration: 0
}) {
    const [recorder, setRecorder] = useState<AudioRecorder | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
    const [lastBlob, setLastBlob] = useState<Blob | null>(null);
    const [playing, setPlaying] = useState(false);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState<string>("Ready to record");
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [visualization, setVisualization] = useState<any | null>(null);
    const [currentResult, setCurrentResult] = useState<any | null>(null); // Store the current result for reprocessing
    const [startTime, setStartTime] = useState<number>(0); // Start time in song for placing notes
    const waveformContainerRef = useRef<HTMLDivElement | null>(null);

    // Handle drum type corrections from visualization modal
    const handleDrumCorrections = (editedHits: any[]) => {
        console.log("[RecorderControls] Applying drum corrections - editedHits count:", editedHits.length);

        if (!currentResult) {
            console.error("[RecorderControls] ERROR: No currentResult available");
            return;
        }

        if (!currentResult.ir) {
            console.error("[RecorderControls] ERROR: No IR in currentResult");
            return;
        }

        // Reconstruct IR with corrected drum types
        const newIR = {
            ...currentResult.ir,
            tracks: currentResult.ir.tracks.map((track: any) => {
                if (track.samples) {
                    // Update samples with corrected drum types
                    const newSamples = editedHits.map(hit => ({
                        sample: hit.drum_type,
                        start: hit.time
                    }));
                    console.log("[RecorderControls] Updating track samples:", newSamples.length, "samples");
                    return {
                        ...track,
                        samples: newSamples
                    };
                }
                return track;
            })
        };

        // Create new result with corrected IR (remove visualization to signal we're done with tuning)
        const correctedResult = {
            ir: newIR,
            metadata: currentResult.metadata
        };

        console.log("[RecorderControls] Calling onMelodyGenerated with corrected result");

        // Always call the callback
        if (onMelodyGenerated) {
            onMelodyGenerated(correctedResult);
        } else {
            console.error("[RecorderControls] ERROR: onMelodyGenerated callback not provided!");
        }

        // Clear visualization state
        setVisualization(null);
        setCurrentResult(null);
    };

    useEffect(() => {
        return () => {
            recorder?.stopRecording();
        };
    }, [recorder]);

    const start = async () => {
        setBusy(true);
        setStatus("Starting microphone...");
        const r = new AudioRecorder({
            onStart: () => {
                setIsRecording(true);
                setBusy(false);
                setStatus("Recording...");
            },
            onStop: (finalBlob) => {
                setLastBlob(finalBlob);
                setIsRecording(false);
                setStatus("Recording complete");
            },
            onError: (err) => {
                console.error(err);
                setBusy(false);
                setIsRecording(false);
                setStatus(`Error: ${err.message}`);
            },
        });
        try {
            await r.startRecording();
            setRecorder(r);
            setAnalyser(r.getAnalyser());
        } catch (err) {
            console.error("start recording failed", err);
            setBusy(false);
            setStatus("Failed to start recording");
        }
    };

    const stop = () => {
        recorder?.stopRecording();
        setAnalyser(null);
        setRecorder(null);
    };

    const play = () => {
        if (!lastBlob) return;
        const url = URL.createObjectURL(lastBlob);
        if (audioRef.current) {
            audioRef.current.src = url;
            audioRef.current.play();
            setPlaying(true);
            setStatus("Playing recording...");
            audioRef.current.onended = () => {
                setPlaying(false);
                setStatus("Playback complete");
            };
        }
    };

    const sendToModel = async () => {
        if (!lastBlob) return;
        setBusy(true);
        setStatus("Processing audio...");

        try {
            // Convert blob to WAV format for best compatibility
            const { wav } = await processAudioBlob(lastBlob, 16000);

            // Create form data with visualization request and start time
            const formData = new FormData();
            formData.append("audio", wav, "recording.wav");
            formData.append("save_training_data", "true");
            formData.append("return_visualization", "true");
            formData.append("start_time", startTime.toString()); // Add start time offset

            setStatus("Sending to model...");

            // Call the appropriate endpoint based on mode
            const endpoint = mode === 'drums' ? '/beatbox2drums' : '/hum2melody';
            const response = await api(endpoint, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }

            const result = await response.json();

            console.log("[RecorderControls] Model result:", result);
            console.log("[RecorderControls] Has visualization?", !!result.visualization);
            console.log("[RecorderControls] Has session_id?", !!result.session_id);
            console.log("[RecorderControls] Has ir?", !!result.ir);

            const noteCount = result.metadata?.num_notes || result.metadata?.num_samples || result.visualization?.segments?.length || 0;
            setStatus(`Generated ${noteCount} ${mode === 'drums' ? 'hits' : 'notes'} at ${startTime.toFixed(1)}s`);

            // Show visualization modal for drums if available
            if (mode === 'drums' && result.visualization) {
                console.log("[RecorderControls] Setting drum visualization data");
                setVisualization(result.visualization);
                setCurrentResult(result); // Store result for later correction
            } else {
                // For melody mode, callback immediately
                if (onMelodyGenerated) {
                    console.log("[RecorderControls] Calling onMelodyGenerated with result");
                    onMelodyGenerated(result);
                }
            }

            setBusy(false);

        } catch (err) {
            console.error("Failed to process audio:", err);
            setStatus(`Failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
            setBusy(false);
        }
    };

    // Format time as MM:SS
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="space-y-5">
            {/* Start Time Selection */}
            {maxDuration > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <Clock className="w-4 h-4 text-purple-400" />
                        <label className="text-sm font-medium text-gray-300">
                            Start Time in Song
                        </label>
                    </div>
                    <div className="space-y-2">
                        <input
                            type="range"
                            min="0"
                            max={maxDuration}
                            step="0.1"
                            value={startTime}
                            onChange={(e) => setStartTime(parseFloat(e.target.value))}
                            disabled={isRecording || busy}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                                     [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                                     [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                                     [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:cursor-pointer
                                     [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4
                                     [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-purple-500
                                     [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer
                                     disabled:opacity-50 disabled:cursor-not-allowed"
                        />
                        <div className="flex justify-between text-xs text-gray-400">
                            <span>0:00</span>
                            <span className="text-purple-400 font-semibold">{formatTime(startTime)}</span>
                            <span>{formatTime(maxDuration)}</span>
                        </div>
                        <p className="text-xs text-gray-500">
                            Generated notes will be placed starting at this time
                        </p>
                    </div>
                </div>
            )}

            {/* Recording Section */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                    <Mic className="w-4 h-4 text-red-400" />
                    <label className="text-sm font-medium text-gray-300">
                        Record {mode === 'drums' ? 'Beatbox' : 'Hum'}
                    </label>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={start}
                        disabled={isRecording || busy}
                        className={`flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2
                                  ${isRecording
                                    ? 'bg-red-600/20 border border-red-500 text-red-400 cursor-not-allowed'
                                    : 'bg-red-600 hover:bg-red-500 border border-red-600 text-white'
                                  }
                                  disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        <Mic className="w-4 h-4" />
                        {isRecording ? "Recording..." : "Start Recording"}
                    </button>

                    <button
                        onClick={stop}
                        disabled={!isRecording}
                        className="px-4 py-2.5 rounded-lg bg-gray-700 hover:bg-gray-600 border border-gray-600 text-white font-medium
                                 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                    >
                        <Square className="w-4 h-4" />
                        Stop
                    </button>
                </div>

                {/* Waveform */}
                <div className="mt-3" ref={waveformContainerRef}>
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700/50 overflow-hidden">
                        <WaveformCanvas
                            analyser={analyser}
                            height={60}
                            width={waveformContainerRef.current?.clientWidth || 400}
                        />
                    </div>
                </div>
            </div>

            {/* Playback Section */}
            {lastBlob && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                    <button
                        onClick={play}
                        disabled={playing || busy}
                        className="w-full px-4 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 border border-blue-600 text-white font-medium
                                 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                    >
                        <Play className="w-4 h-4" />
                        {playing ? "Playing..." : "Play Recording"}
                    </button>
                </div>
            )}

            {/* Status Display */}
            <div className="text-sm text-center">
                <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                                ${busy ? 'bg-purple-600/20 text-purple-300' :
                                  isRecording ? 'bg-red-600/20 text-red-300' :
                                  lastBlob ? 'bg-green-600/20 text-green-300' :
                                  'bg-gray-700/50 text-gray-400'}`}>
                    {busy && <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />}
                    {isRecording && <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse" />}
                    {status}
                </span>
            </div>

            {/* Generate Button */}
            {lastBlob && !busy && (
                <button
                    onClick={sendToModel}
                    disabled={!lastBlob || busy}
                    className="w-full px-6 py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600
                             hover:from-purple-500 hover:to-pink-500 text-white font-semibold text-lg
                             disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02]
                             flex items-center justify-center gap-2 shadow-lg"
                >
                    <Sparkles className="w-5 h-5" />
                    {mode === 'drums' ? "Generate Drum Track" : "Generate Melody"}
                </button>
            )}

            {/* Hidden audio element for playback */}
            <audio ref={audioRef} style={{ display: "none" }} />

            {/* Drum Onset Visualization Modal */}
            {visualization && mode === 'drums' && (
                <DrumOnsetVisualizer
                    visualization={visualization}
                    onClose={() => {
                        console.log("[RecorderControls] Modal closed/cancelled - NOT creating track");
                        setVisualization(null);
                        setCurrentResult(null);
                        if (onVisualizationClose) {
                            onVisualizationClose();
                        }
                    }}
                    onApply={handleDrumCorrections}
                />
            )}
        </div>
    );
}