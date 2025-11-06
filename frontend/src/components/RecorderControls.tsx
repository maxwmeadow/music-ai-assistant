// src/components/RecorderControls.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { AudioRecorder } from "@/lib/audioRecorder";
import { WaveformCanvas } from "./WaveformCanvas";
import { processAudioBlob } from "@/lib/audioProcessing";
import { api } from "@/lib/api";

interface RecorderControlsProps {
    onMelodyGenerated?: (ir: any) => void;
}

export function RecorderControls({ onMelodyGenerated }: RecorderControlsProps = { onMelodyGenerated: undefined }) {
    const [recorder, setRecorder] = useState<AudioRecorder | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
    const [lastBlob, setLastBlob] = useState<Blob | null>(null);
    const [playing, setPlaying] = useState(false);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState<string>("Idle");
    const audioRef = useRef<HTMLAudioElement | null>(null);

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

            // Create form data with visualization request
            const formData = new FormData();
            formData.append("audio", wav, "recording.wav");
            formData.append("save_training_data", "true");
            formData.append("return_visualization", "true");

            setStatus("Sending to model...");

            // Call the /hum2melody endpoint
            const response = await api("/hum2melody", {
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

            const noteCount = result.metadata?.num_notes || result.visualization?.segments?.length || 0;
            setStatus(`Generated ${noteCount} notes (${result.metadata?.model_used || 'unknown'} model)`);

            // Callback with the full result (includes IR, visualization, session_id)
            if (onMelodyGenerated) {
                console.log("[RecorderControls] Calling onMelodyGenerated with result");
                onMelodyGenerated(result);
            }

            setBusy(false);

        } catch (err) {
            console.error("Failed to process audio:", err);
            setStatus(`Failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
            setBusy(false);
        }
    };

    return (
        <div className="space-y-4">
            {/* Controls */}
            <div className="flex items-center gap-2 flex-wrap">
                <button
                    onClick={start}
                    disabled={isRecording || busy}
                    className="px-4 py-2 rounded-lg border bg-red-600 hover:bg-red-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {isRecording ? "⏺ Recording..." : "Start Recording"}
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
                    className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {busy ? "Processing..." : "🎵 Generate Melody"}
                </button>
            </div>

            {/* Status */}
            <div className="text-sm text-gray-300 font-medium px-2">
                Status: {status}
            </div>

            {/* Waveform */}
            <div className="w-full">
                <div className="bg-gray-800 rounded-lg p-3 border border-white/10">
                    <WaveformCanvas analyser={analyser} height={80} width={600} />
                </div>
            </div>

            {/* Hidden audio element for playback */}
            <audio ref={audioRef} style={{ display: "none" }} />
        </div>
    );
}