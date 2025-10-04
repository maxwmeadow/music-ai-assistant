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

  // Function to display toast messages
  const showToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 2000);
  };

  // Fetch test Sonic Pi code from /test
  const fetchTest = async () => {
    setLoadingTest(true);
    try {
      const response = await api("/test");
      const data = await response.json();
      setCode(data.code || "// No code received");
      showToast("Loaded sample code");
    } catch (error) {
      console.error(error);
      showToast("Failed to fetch test code");
    } finally {
      setLoadingTest(false);
    }
  };

  // Send the current code to /run to prepare it for Sonic Pi
  const sendToRunner = async () => {
    setLoadingRun(true);
    try {
      const response = await api("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });

      const data = await response.json();
      setExecutableCode(data.code || "");
      showToast("Code ready to run");
    } catch (error) {
      console.error(error);
      showToast("Failed to prepare code");
    } finally {
      setLoadingRun(false);
    }
  };

  // Send the prepared code to /play to actually execute in Sonic Pi
  const playInSonicPi = async () => {
    if (!executableCode) {
      showToast("Please run the code first");
      return;
    }

    setLoadingPlay(true);
    try {
      const response = await api("/play", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: executableCode }),
      });

      if (response.ok) {
        showToast("Playing in Sonic Pi 🎵");
      } else {
        showToast("Failed to play in Sonic Pi");
      }
    } catch (error) {
      console.error(error);
      showToast("Playback error");
    } finally {
      setLoadingPlay(false);
    }
  };

  return (
      <div className="min-h-screen bg-gray-50 p-8">
        {/* Toast Notification */}
        {toast && (
            <div className="fixed top-4 right-4 bg-gray-800 text-white px-4 py-2 rounded-md shadow-md z-50 animate-fade-in">
              {toast}
            </div>
        )}

        <div className="grid grid-cols-1 font-bold text-black md:grid-cols-2 gap-6 max-w-6xl mx-auto">
          {/* Left Pane: AI Assistant */}
          <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-6">
            <h2 className="text-2xl font-bold">AI Assistant</h2>

            {/* Generate sample code */}
            <div className="flex gap-2">
              <button
                  disabled={loadingTest}
                  onClick={fetchTest}
                  className="flex-1 rounded-md border px-3 py-2 text-sm hover:bg-gray-100 disabled:opacity-50"
              >
                {loadingTest ? "Loading..." : "Generate (/test)"}
              </button>
            </div>

            {/* Music options */}
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

            {/* Audio Recorder Section */}
            <div className="border-t pt-4">
              <h3 className="text-lg font-semibold mb-2">Record Audio</h3>
              <RecorderControls
                  onProcessedAudio={async (processedAudio) => {
                    showToast("Uploading audio...");
                    try {
                      const res = await api("/process-audio", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ audio: processedAudio }),
                      });

                      if (!res.ok) throw new Error("Upload failed");
                      showToast("Audio uploaded successfully");
                    } catch (err) {
                      console.error(err);
                      showToast("Audio upload failed");
                    }
                  }}
              />
            </div>
          </div>

          {/* Right Pane: Code Editor */}
          <div className="bg-white rounded-2xl shadow p-6 flex flex-col gap-4">
            <h2 className="text-2xl font-bold">Generated Sonic Pi Code</h2>

            <CodeEditor code={code} onChange={setCode} />

            {/* Control buttons */}
            <div className="flex flex-wrap gap-3">
              <button
                  disabled={loadingRun}
                  onClick={sendToRunner}
                  className="rounded-md bg-blue-600 text-white px-4 py-2 hover:bg-blue-700 disabled:opacity-50"
              >
                {loadingRun ? "Processing..." : "Run (/run)"}
              </button>

              <button
                  disabled={loadingPlay}
                  onClick={playInSonicPi}
                  className="rounded-md bg-green-600 text-white px-4 py-2 hover:bg-green-700 disabled:opacity-50"
              >
                {loadingPlay ? "Playing..." : "Play in Sonic Pi (/play)"}
              </button>
            </div>

            {/* Executable Code preview */}
            {executableCode && (
                <div className="mt-4 p-3 border rounded-md bg-gray-50 text-sm font-mono whitespace-pre-wrap">
                  {executableCode}
                </div>
            )}
          </div>
        </div>
      </div>
  );
}
