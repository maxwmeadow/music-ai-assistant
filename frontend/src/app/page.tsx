"use client";

import { useState } from "react";
import { CodeEditor } from "@/components/CodeEditor";
import { api } from "@/lib/api";
import { RecorderControls } from "@/components/RecorderControls";

export default function Home() {
  const [code, setCode] = useState("// Sonic Pi code will appear here...");
  const [loadingTest, setLoadingTest] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingPlay, setLoadingPlay] = useState(false);
  const [executableCode, setExecutableCode] = useState<string>("");
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 2000);
  };

  const fetchTest = async () => {
    setLoadingTest(true);
    try {
      const response = await api("/test");
      const dslCode = await response.text();
      setCode(dslCode);
      showToast("Loaded sample code");
    } catch (error) {
      console.error(error);
      showToast("Failed to fetch test code");
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
      showToast("Code ready to run");
    } catch (error) {
      console.error(error);
      showToast("Failed to prepare code");
    } finally {
      setLoadingRun(false);
    }
  };

  const playAudio = async () => {
    if (!executableCode) {
      showToast("Please run the code first");
      return;
    }

    setLoadingPlay(true);
    try {
      const Tone = await import('tone');
      (window as any).Tone = Tone;

      showToast("Playing audio...");
      eval(executableCode);
    } catch (error) {
      console.error(error);
      showToast("Playback error");
    } finally {
      setLoadingPlay(false);
      setTimeout(() => setToast(null), 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {toast && (
        <div className="fixed top-4 right-4 bg-gray-800 text-white px-4 py-2 rounded-md shadow-md z-50 animate-fade-in">
          {toast}
        </div>
      )}

      <div className="grid grid-cols-1 font-bold text-black md:grid-cols-2 gap-6 max-w-6xl mx-auto">
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

          <div className="border-t pt-4">
            <h3 className="text-lg font-semibold mb-2">Record Audio</h3>
            <RecorderControls />
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-4">
          <h2 className="text-2xl font-bold">Generated Sonic Pi Code</h2>

          <CodeEditor value={code} onChange={setCode} />

          <div className="flex flex-wrap gap-3">
            <button
              disabled={loadingRun}
              onClick={sendToRunner}
              className="rounded-md bg-blue-600 text-white px-4 py-2 hover:bg-blue-700 disabled:opacity-50"
            >
              {loadingRun ? "Processing..." : "Run (/run)"}
            </button>

            <button
              disabled={loadingPlay || !executableCode}
              onClick={playAudio}
              className="rounded-md bg-green-600 text-white px-4 py-2 hover:bg-green-700 disabled:opacity-50"
            >
              {loadingPlay ? "Playing..." : "Play"}
            </button>
          </div>

          {executableCode && (
            <div className="mt-4 p-3 border rounded-md bg-gray-50 text-sm font-mono whitespace-pre-wrap overflow-auto max-h-40">
              {executableCode.substring(0, 500)}...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}