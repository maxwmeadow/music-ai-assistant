// src/components/RecorderControls.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { AudioRecorder } from "@/lib/audioRecorder";
import { WaveformCanvas } from "./WaveformCanvas";
import { processAudioBlob } from "@/lib/audioProcessing";
import { api } from "@/lib/api";
import DrumOnsetVisualizer from "./DrumOnsetVisualizer";
import { Spinner } from "./Spinner";

interface RecorderControlsProps {
    onMelodyGenerated?: (ir: any) => void;
    mode?: 'melody' | 'drums';
    onVisualizationClose?: () => void;
}

export function RecorderControls({ onMelodyGenerated, mode = 'melody', onVisualizationClose }: RecorderControlsProps = { onMelodyGenerated: undefined, mode: 'melody', onVisualizationClose: undefined }) {
    const [recorder, setRecorder] = useState<AudioRecorder | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
    const [lastBlob, setLastBlob] = useState<Blob | null>(null);
    const [playing, setPlaying] = useState(false);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState<string>("Idle");
    const [error, setError] = useState<string | null>(null);
    const [recordingDuration, setRecordingDuration] = useState<number>(0);
    const [permissionState, setPermissionState] = useState<PermissionState | null>(null);
    const [isSupported, setIsSupported] = useState<boolean>(true);
    const [streamActive, setStreamActive] = useState<boolean>(true);
    const [isClipping, setIsClipping] = useState<boolean>(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [visualization, setVisualization] = useState<any | null>(null);
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Check browser support and permission on mount
    useEffect(() => {
        const checkSupport = async () => {
            setIsSupported(AudioRecorder.isSupported());
            if (AudioRecorder.isSupported()) {
                const perm = await AudioRecorder.checkPermission();
                setPermissionState(perm);
            }
        };
        checkSupport();
    }, []);

    useEffect(() => {
        return () => {
            recorder?.stopRecording();
        };
    }, [recorder]);

    const formatDuration = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const start = async () => {
        setError(null);
        setBusy(true);
        setStatus("Checking microphone access...");
        
        // Check browser support
        if (!AudioRecorder.isSupported()) {
            setError("Your browser doesn't support audio recording. Please use Chrome, Firefox, Edge, or Safari (14.1+).");
            setBusy(false);
            setStatus("Browser not supported");
            return;
        }

        // Check permission state
        const perm = await AudioRecorder.checkPermission();
        setPermissionState(perm);
        
        if (perm === 'denied') {
            setError("Microphone permission denied. Please allow microphone access in your browser settings and try again.");
            setBusy(false);
            setStatus("Permission denied");
            return;
        }

        setStatus("Requesting microphone access...");
        const r = new AudioRecorder({
            onStart: () => {
                setIsRecording(true);
                setBusy(false);
                setStatus("Recording...");
                setRecordingDuration(0);
                setError(null);
            },
            onStop: (finalBlob) => {
                setLastBlob(finalBlob);
                setIsRecording(false);
                const duration = recordingDuration;
                setStatus(`Recording complete (${formatDuration(duration)})`);
            },
            onError: (err) => {
                console.error("Recording error:", err);
                setBusy(false);
                setIsRecording(false);
                setError(err.message);
                setStatus(`Error: ${err.message}`);
            },
            onStreamActive: (active) => {
                setStreamActive(active);
                if (!active && isRecording) {
                    setError("Microphone stream stopped. Check your microphone connection.");
                }
            },
            onDurationUpdate: (duration) => {
                setRecordingDuration(duration);
            },
            onClippingDetected: (clipping) => {
                setIsClipping(clipping);
            },
        });
        try {
            await r.startRecording();
            setRecorder(r);
            setAnalyser(r.getAnalyser());
            setPermissionState('granted');
        } catch (err: any) {
            console.error("start recording failed", err);
            setBusy(false);
            setError(err.message || "Failed to start recording");
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
        setError(null);
        setStatus("Processing audio...");

        // Clear any existing timeout
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }

        // Set timeout for audio processing
        timeoutRef.current = setTimeout(() => {
            setError("Processing took too long. Try a shorter recording (max 30 seconds).");
            setBusy(false);
            setStatus("Processing timeout");
            timeoutRef.current = null;
        }, 30000); // 30 second timeout

        try {
            // Convert blob to WAV format for best compatibility
            setStatus("Converting audio format...");
            const { wav } = await processAudioBlob(lastBlob, 16000);

            // Create form data with visualization request
            const formData = new FormData();
            formData.append("audio", wav, "recording.wav");
            formData.append("save_training_data", "true");
            formData.append("return_visualization", "true");

            setStatus("Sending to model...");

            // Call the appropriate endpoint based on mode
            const endpoint = mode === 'drums' ? '/beatbox2drums' : '/hum2melody';
            const response = await api(endpoint, {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            console.log("[RecorderControls] Model result:", result);
            console.log("[RecorderControls] Has visualization?", !!result.visualization);
            console.log("[RecorderControls] Has session_id?", !!result.session_id);
            console.log("[RecorderControls] Has ir?", !!result.ir);

            const noteCount = result.metadata?.num_notes || result.metadata?.num_samples || result.visualization?.segments?.length || 0;
            setStatus(`Generated ${noteCount} ${mode === 'drums' ? 'hits' : 'notes'} (${result.metadata?.model_used || 'unknown'} model)`);

            // Show visualization modal for drums if available
            if (mode === 'drums' && result.visualization) {
                console.log("[RecorderControls] Setting drum visualization data");
                setVisualization(result.visualization);
            }

            // Callback with the full result (includes IR, visualization, session_id)
            if (onMelodyGenerated) {
                console.log("[RecorderControls] Calling onMelodyGenerated with result");
                onMelodyGenerated(result);
            }

            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
            }
            setBusy(false);

        } catch (err: any) {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
            }
            console.error("Failed to process audio:", err);
            
            // Use error message from AppError if available
            const errorMessage = err?.userMessage || err?.message || 'Unknown error';
            setError(errorMessage);
            setStatus(`Failed: ${errorMessage}`);
            setBusy(false);
        }
    };

    return (
        <div className="space-y-4">
            {/* Browser Support Warning */}
            {!isSupported && (
                <div className="bg-yellow-900/30 border border-yellow-600 rounded-lg p-3">
                    <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                        <div>
                            <p className="text-yellow-400 font-medium text-sm">Browser Not Supported</p>
                            <p className="text-yellow-300 text-xs mt-1">
                                Your browser doesn't support audio recording. Please use Chrome, Firefox, Edge, or Safari (14.1+).
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Permission Denied Warning */}
            {permissionState === 'denied' && (
                <div className="bg-red-900/30 border border-red-600 rounded-lg p-3">
                    <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                        <div>
                            <p className="text-red-400 font-medium text-sm">Microphone Permission Denied</p>
                            <p className="text-red-300 text-xs mt-1">
                                Please allow microphone access in your browser settings. Look for the microphone icon in your browser's address bar.
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Error Message */}
            {error && (
                <div className="bg-red-900/30 border border-red-600 rounded-lg p-3">
                    <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                        <div>
                            <p className="text-red-400 font-medium text-sm">Error</p>
                            <p className="text-red-300 text-xs mt-1">{error}</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Stream Inactive Warning */}
            {isRecording && !streamActive && (
                <div className="bg-orange-900/30 border border-orange-600 rounded-lg p-3">
                    <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-orange-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                        <div>
                            <p className="text-orange-400 font-medium text-sm">Microphone Disconnected</p>
                            <p className="text-orange-300 text-xs mt-1">
                                Check your microphone connection. Recording may be interrupted.
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Clipping Warning */}
            {isRecording && isClipping && (
                <div className="bg-yellow-900/30 border border-yellow-600 rounded-lg p-3">
                    <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                        <div>
                            <p className="text-yellow-400 font-medium text-sm">Audio Clipping Detected</p>
                            <p className="text-yellow-300 text-xs mt-1">
                                Input level is too high. Lower your microphone volume or move further from the mic to avoid distortion.
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Controls */}
            <div className="flex items-center gap-2 flex-wrap">
                <button
                    onClick={start}
                    disabled={isRecording || busy || !isSupported || permissionState === 'denied'}
                    className="px-4 py-2 rounded-lg border bg-red-600 hover:bg-red-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                    {isRecording ? (
                        <>
                            <span className="animate-pulse">⏺</span>
                            Recording... {formatDuration(recordingDuration)}
                        </>
                    ) : (
                        <>
                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                            </svg>
                            Start Recording
                        </>
                    )}
                </button>

                <button
                    onClick={stop}
                    disabled={!isRecording}
                    className="px-4 py-2 rounded-lg border bg-gray-700 hover:bg-gray-600 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    Stop
                </button>

                <button
                    onClick={play}
                    disabled={!lastBlob || playing}
                    className="px-4 py-2 rounded-lg border bg-blue-600 hover:bg-blue-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {playing ? "▶ Playing..." : "Play Recording"}
                </button>

                <button
                    onClick={sendToModel}
                    disabled={!lastBlob || busy}
                    className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                    {busy ? (
                        <>
                            <Spinner size="sm" />
                            Processing...
                        </>
                    ) : (
                        mode === 'drums' ? "🥁 Generate Drums" : "🎵 Generate Melody"
                    )}
                </button>
            </div>

            {/* Status */}
            <div className="text-sm text-gray-300 font-medium px-2 flex items-center gap-2">
                <span>Status: {status}</span>
                {isRecording && (
                    <span className="text-red-400 font-mono">
                        {formatDuration(recordingDuration)}
                    </span>
                )}
            </div>

            {/* Waveform */}
            <div className="w-full">
                <div className="bg-gray-800 rounded-lg p-3 border border-white/10">
                    <WaveformCanvas analyser={analyser} height={80} width={600} />
                </div>
            </div>

            {/* Hidden audio element for playback */}
            <audio ref={audioRef} style={{ display: "none" }} />

            {/* Drum Onset Visualization Modal */}
            {visualization && mode === 'drums' && (
                <DrumOnsetVisualizer
                    visualization={visualization}
                    onClose={() => {
                        setVisualization(null);
                        // Notify parent to close the recorder modal too
                        if (onVisualizationClose) {
                            onVisualizationClose();
                        }
                    }}
                />
            )}
        </div>
    );
}