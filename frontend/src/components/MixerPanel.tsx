"use client";

import { useState, useEffect } from "react";
import { ParsedTrack } from "@/lib/dslParser";
import { Music } from "lucide-react";

interface MixerPanelProps {
  tracks: ParsedTrack[];
  onVolumeChange: (trackId: string, volume: number) => void;
}

export function MixerPanel({ tracks, onVolumeChange }: MixerPanelProps) {
  const [volumes, setVolumes] = useState<Record<string, number>>({});
  const [mutes, setMutes] = useState<Record<string, boolean>>({});
  const [solos, setSolos] = useState<Record<string, boolean>>({});

  // Initialize volumes
  useEffect(() => {
    const initialVolumes: Record<string, number> = {};
    tracks.forEach(track => {
      initialVolumes[track.id] = 0; // 0 dB
    });
    setVolumes(initialVolumes);
  }, [tracks]);

  const handleVolumeChange = (trackId: string, value: number) => {
    const newVolumes = { ...volumes, [trackId]: value };
    setVolumes(newVolumes);

    // Calculate actual volume considering mute/solo
    const hasSolo = Object.values(solos).some(s => s);
    const isMuted = mutes[trackId];
    const isSoloed = solos[trackId];

    let actualVolume = value;
    if (isMuted || (hasSolo && !isSoloed)) {
      actualVolume = -Infinity; // Mute
    }

    onVolumeChange(trackId, actualVolume);
  };

  const toggleMute = (trackId: string) => {
    const newMutes = { ...mutes, [trackId]: !mutes[trackId] };
    setMutes(newMutes);

    // Recalculate actual volume
    const hasSolo = Object.values(solos).some(s => s);
    const actualVolume = newMutes[trackId] || (hasSolo && !solos[trackId])
      ? -Infinity
      : volumes[trackId];

    onVolumeChange(trackId, actualVolume);
  };

  const toggleSolo = (trackId: string) => {
    const newSolos = { ...solos, [trackId]: !solos[trackId] };
    setSolos(newSolos);

    const hasSolo = Object.values(newSolos).some(s => s);

    // Recalculate all volumes
    tracks.forEach(track => {
      const isMuted = mutes[track.id];
      const isSoloed = newSolos[track.id];
      const actualVolume = isMuted || (hasSolo && !isSoloed)
        ? -Infinity
        : volumes[track.id];

      onVolumeChange(track.id, actualVolume);
    });
  };

  if (tracks.length === 0) {
    return (
      <div className="text-sm text-gray-500 p-4">
        No tracks detected. Run code to see mixer.
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <Music className="w-5 h-5" />
        Mixer
      </h3>

      <div className="flex gap-4 overflow-x-auto pb-2">
        {tracks.map((track) => (
          <div key={track.id} className="flex flex-col items-center min-w-[90px] bg-white/5 rounded-xl p-3 border border-white/10">
            {/* Track label */}
            <div className="text-xs font-semibold text-purple-300 mb-3 truncate w-full text-center">
              {track.id}
            </div>

            {/* Volume slider */}
            <div className="h-32 flex items-center justify-center mb-3">
              <input
                type="range"
                min="-40"
                max="12"
                step="1"
                value={volumes[track.id] ?? 0}
                onChange={(e) => handleVolumeChange(track.id, Number(e.target.value))}
                className="w-32 origin-center -rotate-90 appearance-none bg-white/10 rounded-full h-2 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-br [&::-webkit-slider-thumb]:from-purple-500 [&::-webkit-slider-thumb]:to-pink-500 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-lg"
              />
            </div>

            {/* Volume display */}
            <div className="text-xs font-mono text-white mb-3">
              {mutes[track.id] || (Object.values(solos).some(s => s) && !solos[track.id])
                ? "MUTE"
                : `${volumes[track.id] ?? 0}dB`}
            </div>

            {/* Mute/Solo buttons */}
            <div className="flex gap-2 mb-2">
              <button
                onClick={() => toggleMute(track.id)}
                className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${
                  mutes[track.id]
                    ? "bg-red-500 text-white shadow-lg shadow-red-500/50"
                    : "bg-white/10 text-gray-300 hover:bg-white/20"
                }`}
              >
                M
              </button>
              <button
                onClick={() => toggleSolo(track.id)}
                className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${
                  solos[track.id]
                    ? "bg-yellow-500 text-white shadow-lg shadow-yellow-500/50"
                    : "bg-white/10 text-gray-300 hover:bg-white/20"
                }`}
              >
                S
              </button>
            </div>

            {/* Instrument name */}
            <div className="text-xs text-gray-400 truncate w-full text-center">
              {track.instrument?.split('/').pop() || "synth"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}