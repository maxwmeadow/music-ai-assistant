// src/components/RecorderControls.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { AudioRecorder } from "@/lib/audioRecorder";
import { WaveformCanvas } from "./WaveformCanvas";
import { processAudioBlob } from "@/lib/audioProcessing";
import { api } from "@/lib/api";

export function RecorderControls() {
    const [recorder, setRecorder] = useState<AudioRecorder | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
    const [lastBlob, setLastBlob] = useState<Blob | null>(null);
    const [playing, setPlaying] = useState(false);
    const [busy, setBusy] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    useEffect(() => {
        return () => {
            recorder?.stopRecording();
        };
    }, [recorder]);

    const start = async () => {
        setBusy(true);
        const r = new AudioRecorder({
            onStart: () => {
                setIsRecording(true);
                setBusy(false);
            },
            onStop: (finalBlob) => {
                setLastBlob(finalBlob);
                setIsRecording(false);
            },
            onError: (err) => {
                console.error(err);
                setBusy(false);
                setIsRecording(false);
            },
        });
        try {
            await r.startRecording();
            setRecorder(r);
            setAnalyser(r.getAnalyser());
        } catch (err) {
            console.error("start recording failed", err);
            setBusy(false);
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
            audioRef.current.onended = () => setPlaying(false);
        }
    };

    const send = async () => {
        if (!lastBlob) return;
        setBusy(true);
        try {
            const { float32, wav } = await processAudioBlob(lastBlob, 16000);
            // Send WAV as form data (or send float32 as binary payload depending on backend)
            const form = new FormData();
            form.append("file", wav, "recording.wav");
            // or: send JSON with base64 of WAV (if backend prefers)
            const res = await api("/process-audio", {
                method: "POST",
                body: form,
            });
            if (!res.ok) throw new Error("upload failed");
            setBusy(false);
            alert("Audio sent!");
        } catch (err) {
            console.error(err);
            alert("Failed to send audio");
            setBusy(false);
        }
    };

    return (
        <div className="space-y-3">
            <div className="flex items-center gap-2">
                <button
                    onClick={start}
                    disabled={isRecording || busy}
                    className="px-3 py-2 rounded border bg-white disabled:opacity-50"
                >
                    Start
                </button>
                <button
                    onClick={stop}
                    disabled={!isRecording}
                    className="px-3 py-2 rounded border bg-white disabled:opacity-50"
                >
                    Stop
                </button>
                <button onClick={play} disabled={!lastBlob || playing} className="px-3 py-2 rounded border">
                    Play
                </button>
                <button onClick={send} disabled={!lastBlob || busy} className="px-3 py-2 rounded bg-black text-white disabled:opacity-50">
                    Send to model
                </button>
                <div className="ml-2 text-sm text-gray-600">{isRecording ? "Recording…" : lastBlob ? "Ready" : "Idle"}</div>
            </div>

            <div className="w-full">
                <div className="bg-gray-100 rounded p-2">
                    <WaveformCanvas analyser={analyser} height={80} width={500} />
                </div>
            </div>

            <audio ref={audioRef} controls style={{ display: "none" }} />
        </div>
    );
}
