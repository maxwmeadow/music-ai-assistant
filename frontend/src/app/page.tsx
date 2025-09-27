"use client";

import { useState } from "react";
import { CodeEditor } from "@/components/CodeEditor";
import { api } from "@/lib/api";

export default function Home() {
  const [code, setCode] = useState("// Sonic Pi code will appear here...");
  const [loadingTest, setLoadingTest] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingPlay, setLoadingPlay] = useState(false);
  const [executableCode, setExecutableCode] = useState<string>('');
  const [toast, setToast] = useState<string | null>(null);

  const fetchTest = async () => {
    try {
      setLoadingTest(true);
      setToast("Fetching code...");
      const res = await api("/test");
      if (!res.ok) throw new Error("Failed to fetch /test");
      const text = await res.text();
      setCode(text);
      setToast("Loaded from /test");
    } catch {
      setToast("Could not load code");
    } finally {
      setLoadingTest(false);
      setTimeout(() => setToast(null), 2000);
    }
  };

  const sendToRunner = async () => {
    try {
      setLoadingRun(true);
      setToast("Sending to runner...");
      const res = await api("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) throw new Error("Runner error");

      // Get the response and extract executable code
      const result = await res.json();
      console.log("Runner response:", result);

      if (result.meta && result.meta.executable_code) {
        setExecutableCode(result.meta.executable_code);
        setToast("Ready to play!");
      } else {
        setToast("Sent successfully");
      }
    } catch {
      setToast("Failed to send");
    } finally {
      setLoadingRun(false);
      setTimeout(() => setToast(null), 2000);
    }
  };

  const playAudio = async () => {
    try {
      setLoadingPlay(true);

      if (!executableCode) {
        setToast("No audio code available");
        return;
      }

      setToast("Loading Tone.js...");

      // Dynamically import Tone.js
      const Tone = await import('tone');

      // Make Tone available globally for the eval'd code
      (window as any).Tone = Tone;

      setToast("Playing audio...");

      // Execute the Tone.js code
      eval(executableCode);

    } catch (error) {
      console.error("Audio error:", error);
      setToast("Audio playback failed");
    } finally {
      setLoadingPlay(false);
      setTimeout(() => setToast(null), 3000);
    }
  };

  const stopAudio = async () => {
    try {
      const Tone = await import('tone');
      Tone.Transport.stop();
      Tone.Transport.cancel();
      setToast("⏹️ Audio stopped");
      setTimeout(() => setToast(null), 2000);
    } catch (error) {
      console.error("Stop audio error:", error);
    }
  };

  return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-6xl mx-auto">
          {/* Left Pane: Assistant */}
          <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-6">
            <h2 className="text-2xl font-bold">AI Assistant</h2>

            <div className="flex gap-2">
              <button
                  disabled={loadingTest}
                  onClick={fetchTest}
                  className="flex-1 rounded-md border px-3 py-2 text-sm hover:bg-gray-100 disabled:opacity-50"
              >
                {loadingTest ? "Loading..." : "Generate (/test)"}
              </button>
            </div>

            {/* Options */}
            <div className="flex flex-col gap-3">
              <button className="w-full text-left rounded-lg border px-4 py-3 hover:bg-gray-50">
                <p className="font-semibold">Hum a melody</p>
                <p className="text-sm text-gray-600">
                  I'll generate a melody based on your humming.
                </p>
              </button>
              <button className="w-full text-left rounded-lg border px-4 py-3 hover:bg-gray-50">
                <p className="font-semibold">Beatbox a drum pattern</p>
                <p className="text-sm text-gray-600">
                  Here's a drum pattern inspired by your beatboxing.
                </p>
              </button>
              <button className="w-full text-left rounded-lg border px-4 py-3 hover:bg-gray-50">
                <p className="font-semibold">Add a bassline</p>
                <p className="text-sm text-gray-600">
                  I'll create a bassline to go with your melody.
                </p>
              </button>
            </div>
          </div>

          {/* Right Pane: Code Editor */}
          <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-4 relative">
            <h2 className="text-2xl font-bold">Sonic Pi Code</h2>

            <div className="flex-1 min-h-[400px]">
              <CodeEditor value={code} onChange={setCode} />
            </div>

            {/* Control Buttons */}
            <div className="flex gap-2 justify-end">
              <button
                  onClick={sendToRunner}
                  disabled={loadingRun}
                  className="rounded-md bg-black text-white px-4 py-2 hover:bg-gray-800 disabled:opacity-50"
              >
                {loadingRun ? "Sending..." : "Send to Runner"}
              </button>

              {executableCode && (
                <>
                  <button
                      onClick={playAudio}
                      disabled={loadingPlay}
                      className="rounded-md bg-green-600 text-white px-4 py-2 hover:bg-green-700 disabled:opacity-50"
                  >
                    {loadingPlay ? "Loading..." : "▶️ Play"}
                  </button>

                  <button
                      onClick={stopAudio}
                      className="rounded-md bg-red-600 text-white px-4 py-2 hover:bg-red-700"
                  >
                    ⏹️ Stop
                  </button>
                </>
              )}
            </div>

            {/* Toast */}
            {toast && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-sm px-4 py-2 rounded shadow z-10">
                  {toast}
                </div>
            )}
          </div>
        </div>
      </div>
  );
}