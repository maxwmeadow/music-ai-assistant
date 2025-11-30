"use client";

import React, { useState, useEffect, memo } from "react";
import { ParsedTrack } from "@/lib/dslParser";
import { Music } from "lucide-react";

interface MixerPanelProps {
  tracks: ParsedTrack[];
  trackVolumes: Record<string, number>;
  trackPans: Record<string, number>;
  trackMutes: Record<string, boolean>;
  trackSolos: Record<string, boolean>;
  masterVolume: number;
  masterPan: number;
  onVolumeChange: (trackId: string, volume: number) => void;
  onPanChange: (trackId: string, pan: number) => void;
  onMuteChange: (trackId: string, muted: boolean) => void;
  onSoloChange: (trackId: string, soloed: boolean) => void;
  onMasterVolumeChange: (volume: number) => void;
  onMasterPanChange: (pan: number) => void;
}

export const MixerPanel = memo(function MixerPanel({
  tracks,
  trackVolumes,
  trackPans,
  trackMutes,
  trackSolos,
  masterVolume,
  masterPan,
  onVolumeChange,
  onPanChange,
  onMuteChange,
  onSoloChange,
  onMasterVolumeChange,
  onMasterPanChange
}: MixerPanelProps) {
  // Only keep state for meters and levels (visualization only)
  const [levels, setLevels] = useState<Record<string, number>>({});
  const [meters, setMeters] = useState<Record<string, any>>({});
  const [masterLevel, setMasterLevel] = useState<number>(-60);
  const [masterMeter, setMasterMeter] = useState<any>(null);

  // Create meters and apply panning to all voices when pools are ready
  useEffect(() => {
    const checkInterval = setInterval(async () => {
      if ((window as any).__musicControls?.pools && (window as any).Tone) {
        const pools = (window as any).__musicControls.pools;
        const Tone = (window as any).Tone;
        const newMeters: Record<string, any> = { ...meters };

        pools.forEach((pool: any, trackId: string) => {
          if (!newMeters[trackId] && pool?.voices && pool.voices[0]) {
            // Create a Meter attached to each voice
            const meter = new Tone.Meter();
            pool.voices.forEach((voice: any) => {
              voice.connect(meter);
            });
            newMeters[trackId] = meter;
          }

          // Apply panning to all voices for this track
          const panValue = trackPans[trackId] ?? 0;
          if (pool?.voices) {
            pool.voices.forEach((voice: any) => {
              if (!voice.__panner && voice.disconnect) {
                // Create panner for this voice
                const panner = new Tone.Panner(panValue / 100);
                voice.disconnect();
                voice.connect(panner);
                panner.toDestination();
                voice.__panner = panner;

                // Reconnect meter after panning (voice.disconnect() removed it)
                if (newMeters[trackId]) {
                  voice.connect(newMeters[trackId]);
                }
              } else if (voice.__panner) {
                // Update existing panner
                voice.__panner.pan.value = panValue / 100;
              }
            });
          }
        });

        if (Object.keys(newMeters).length > Object.keys(meters).length) {
          setMeters(newMeters);
        }
      }
    }, 100);

    return () => clearInterval(checkInterval);
  }, [meters, trackPans]);

  // Create master meter on Tone.Destination
  useEffect(() => {
    const checkInterval = setInterval(() => {
      if ((window as any).Tone && !masterMeter) {
        const Tone = (window as any).Tone;
        const meter = new Tone.Meter();
        Tone.Destination.connect(meter);
        setMasterMeter(meter);
      }
    }, 100);

    return () => clearInterval(checkInterval);
  }, [masterMeter]);

  // Apply mute/solo state to audio engine whenever state changes (important for loading saved projects)
  useEffect(() => {
    if ((window as any).__musicControls?.pools) {
      const hasSolo = Object.values(trackSolos).some(s => s);

      tracks.forEach(track => {
        const isMuted = trackMutes[track.id];
        const isSoloed = trackSolos[track.id];
        const actualVolume = isMuted || (hasSolo && !isSoloed)
          ? -Infinity
          : trackVolumes[track.id] ?? 0;

        const pool = (window as any).__musicControls.pools.get(track.id);
        if (pool?.voices) {
          pool.voices.forEach((voice: any) => {
            if (voice.volume) {
              const preGain = voice.__preGain ?? 0;
              voice.volume.value = actualVolume === -Infinity ? -Infinity : (preGain + actualVolume);
            }
          });
        }
      });
    }
  }, [trackMutes, trackSolos, trackVolumes, tracks]);

  // Monitor actual audio levels from meters (tracks + master)
  useEffect(() => {
    const interval = setInterval(() => {
      const newLevels: Record<string, number> = {};

      Object.entries(meters).forEach(([trackId, meter]) => {
        if (meter) {
          // Get the current audio level from the meter
          const value = meter.getValue();

          // Tone.Meter returns values from -Infinity to 0 in dB
          // If it's already in dB, use it directly
          let dbValue: number;

          if (typeof value === 'number') {
            // If value is between 0 and 1, it's linear amplitude - convert to dB
            if (value >= 0 && value <= 1) {
              dbValue = value === 0 ? -Infinity : 20 * Math.log10(value);
            }
            // If value is already negative, it's likely already in dB
            else if (value < 0) {
              dbValue = value;
            }
            // If value is > 1, it's linear amplitude above 0dB
            else {
              dbValue = 20 * Math.log10(value);
            }
          } else {
            dbValue = -Infinity;
          }

          // Clamp to reasonable range
          newLevels[trackId] = Math.max(-60, Math.min(12, dbValue));
        } else {
          newLevels[trackId] = -60;
        }
      });

      setLevels(newLevels);

      // Update master level
      if (masterMeter) {
        const value = masterMeter.getValue();
        let dbValue: number;

        if (typeof value === 'number') {
          if (value >= 0 && value <= 1) {
            dbValue = value === 0 ? -Infinity : 20 * Math.log10(value);
          } else if (value < 0) {
            dbValue = value;
          } else {
            dbValue = 20 * Math.log10(value);
          }
        } else {
          dbValue = -Infinity;
        }

        setMasterLevel(Math.max(-60, Math.min(12, dbValue)));
      }
    }, 50); // Update 20 times per second

    return () => clearInterval(interval);
  }, [meters, masterMeter]);

  const handleVolumeChange = (trackId: string, value: number) => {
    // Call the callback to update parent state
    onVolumeChange(trackId, value);

    // Calculate actual volume considering mute/solo
    const hasSolo = Object.values(trackSolos).some(s => s);
    const isMuted = trackMutes[trackId];
    const isSoloed = trackSolos[trackId];

    let actualVolume = value;
    if (isMuted || (hasSolo && !isSoloed)) {
      actualVolume = -Infinity;
    }

    // Apply volume to each voice, preserving their pre-gain
    if ((window as any).__musicControls?.pools) {
      const pool = (window as any).__musicControls.pools.get(trackId);
      if (pool?.voices) {
        pool.voices.forEach((voice: any) => {
          if (voice.volume) {
            // Get the original pre-gain that was set during instrument creation
            const preGain = voice.__preGain ?? 0;
            // Add user's volume adjustment to the pre-gain
            voice.volume.value = actualVolume === -Infinity ? -Infinity : (preGain + actualVolume);
          }
        });
      }
    }
  };

  const toggleMute = (trackId: string) => {
    const newMuted = !trackMutes[trackId];
    onMuteChange(trackId, newMuted);

    const hasSolo = Object.values(trackSolos).some(s => s);
    const actualVolume = newMuted || (hasSolo && !trackSolos[trackId])
      ? -Infinity
      : trackVolumes[trackId] ?? 0;

    // Apply directly to audio engine without calling onVolumeChange
    if ((window as any).__musicControls?.pools) {
      const pool = (window as any).__musicControls.pools.get(trackId);
      if (pool?.voices) {
        pool.voices.forEach((voice: any) => {
          if (voice.volume) {
            const preGain = voice.__preGain ?? 0;
            voice.volume.value = actualVolume === -Infinity ? -Infinity : (preGain + actualVolume);
          }
        });
      }
    }
  };

  const toggleSolo = (trackId: string) => {
    const newSoloed = !trackSolos[trackId];
    onSoloChange(trackId, newSoloed);

    // Build new solos object with this change
    const newSolos = { ...trackSolos, [trackId]: newSoloed };
    const hasSolo = Object.values(newSolos).some(s => s);

    // Apply volume changes directly to Tone.js without updating parent state
    tracks.forEach(track => {
      const isMuted = trackMutes[track.id];
      const isSoloed = newSolos[track.id];
      const actualVolume = isMuted || (hasSolo && !isSoloed)
        ? -Infinity
        : trackVolumes[track.id] ?? 0;

      // Apply directly to audio engine without calling onVolumeChange
      if ((window as any).__musicControls?.pools) {
        const pool = (window as any).__musicControls.pools.get(track.id);
        if (pool?.voices) {
          pool.voices.forEach((voice: any) => {
            if (voice.volume) {
              const preGain = voice.__preGain ?? 0;
              voice.volume.value = actualVolume === -Infinity ? -Infinity : (preGain + actualVolume);
            }
          });
        }
      }
    });
  };

  const handleLocalPanChange = (trackId: string, value: number) => {
    // Call the callback to update parent state
    onPanChange(trackId, value);

    // Apply pan to each voice in the track
    if ((window as any).__musicControls?.pools) {
      const pool = (window as any).__musicControls.pools.get(trackId);
      if (pool?.voices) {
        pool.voices.forEach((voice: any) => {
          // Create panner if it doesn't exist
          if (!voice.__panner && (window as any).Tone) {
            const Tone = (window as any).Tone;
            const panner = new Tone.Panner(value / 100); // Convert -100..100 to -1..1
            voice.disconnect();
            voice.connect(panner);
            panner.toDestination();
            voice.__panner = panner;
          } else if (voice.__panner) {
            // Update existing panner
            voice.__panner.pan.value = value / 100;
          }
        });
      }
    }
  };

  const handleLocalMasterVolumeChange = (value: number) => {
    // Call the callback to update parent state
    onMasterVolumeChange(value);

    // Apply master volume to Tone.Destination
    if ((window as any).Tone) {
      const Tone = (window as any).Tone;
      Tone.Destination.volume.value = value;
    }
  };

  const handleLocalMasterPanChange = (value: number) => {
    // Call the callback to update parent state
    onMasterPanChange(value);

    // Note: Master pan doesn't make sense for final stereo output
    // This would be for stereo width control instead
    // For now, we'll just store it but not apply it
    // (Real DAWs use this for M/S processing or width control)
  };

  if (tracks.length === 0) {
    return (
      <div className="text-sm text-gray-500 p-4">
        No tracks detected. Run code to see mixer.
      </div>
    );
  }

  return (
    <div className="flex gap-3 pb-2">
      {/* Individual Track Faders - Scrollable */}
      <div className="flex gap-3 overflow-x-auto flex-1">
        {tracks.map((track) => {
        const level = levels[track.id] ?? -60;
        // Convert dB to percentage for meter (-60dB to 0dB range)
        const levelPercent = Math.max(0, Math.min(100, ((level + 60) / 60) * 100));

        return (
          <div key={track.id} className="flex flex-col items-center min-w-[80px] bg-[#2a2a2a] rounded-lg p-1.5 border border-gray-700">
            {/* Track label */}
            <div className="text-[10px] font-semibold text-gray-300 mb-1 truncate w-full text-center">
              {track.id}
            </div>

            {/* Meter + Volume slider container */}
            <div className="h-24 flex items-center gap-2 mb-0.5">
              {/* dB Meter - vertical bar measuring actual audio */}
              <div className="flex flex-col items-center gap-1">
                <div className="w-1.5 h-20 bg-gray-800 rounded-full overflow-hidden relative">
                  <div
                    className={`absolute bottom-0 w-full transition-all duration-75 ${
                      levelPercent > 90 ? 'bg-red-500' :
                      levelPercent > 70 ? 'bg-yellow-500' :
                      'bg-blue-500'
                    }`}
                    style={{ height: `${levelPercent}%` }}
                  />
                  {/* Peak indicators */}
                  {[0, 25, 50, 75, 90].map(threshold => (
                    <div
                      key={threshold}
                      className="absolute w-full h-px bg-gray-600"
                      style={{ bottom: `${threshold}%` }}
                    />
                  ))}
                </div>
                {/* dB readout under meter */}
                <div className="text-[9px] font-mono text-blue-400 w-6 text-center">
                  {level === -Infinity || level <= -60 ? '-∞' : level.toFixed(0)}
                </div>
              </div>

              {/* Volume slider (rotated) */}
              <input
                type="range"
                min="-40"
                max="12"
                step="1"
                value={trackVolumes[track.id] ?? 0}
                onChange={(e) => handleVolumeChange(track.id, Number(e.target.value))}
                className="w-24 origin-center -rotate-90 appearance-none bg-gray-700 rounded-full h-1.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-md"
              />
            </div>

            {/* Volume display and Pan control - combined row */}
            <div className="flex items-center justify-between w-full mb-1 px-1">
              <div className="text-[9px] font-mono text-white">
                {trackMutes[track.id] || (Object.values(trackSolos).some(s => s) && !trackSolos[track.id])
                  ? "MUTE"
                  : `${trackVolumes[track.id] ?? 0}dB`}
              </div>
              <div className="text-[8px] font-mono text-green-400">
                {(trackPans[track.id] ?? 0) === 0 ? 'C' :
                 (trackPans[track.id] ?? 0) < 0 ? `L${Math.abs(trackPans[track.id] ?? 0)}` :
                 `R${trackPans[track.id] ?? 0}`}
              </div>
            </div>

            {/* Pan control slider */}
            <div className="w-full mb-1 px-1">
              <input
                type="range"
                min="-100"
                max="100"
                step="1"
                value={trackPans[track.id] ?? 0}
                onChange={(e) => handleLocalPanChange(track.id, Number(e.target.value))}
                className="w-full appearance-none bg-gray-700 rounded-full h-1 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-green-500 [&::-webkit-slider-thumb]:cursor-pointer"
              />
            </div>

              {/* Mute/Solo buttons */}
              <div className="flex gap-1.5 mb-0.5">
                <button
                  onClick={() => toggleMute(track.id)}
                  className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${
                    trackMutes[track.id]
                      ? "bg-red-500 text-white"
                      : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  M
                </button>
                <button
                  onClick={() => toggleSolo(track.id)}
                  className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${
                    trackSolos[track.id]
                      ? "bg-yellow-500 text-white"
                      : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  S
                </button>
              </div>

              {/* Instrument name */}
              <div className="text-[9px] text-gray-500 truncate w-full text-center">
                {track.instrument?.split('/').pop() || "synth"}
              </div>
            </div>
          );
        })}
      </div>

      {/* Master Fader - Fixed on right, same size as track mixers */}
      <div className="flex-shrink-0 ml-3 border-l-2 border-gray-600 pl-3">
        <div className="flex flex-col items-center min-w-[80px] bg-gradient-to-b from-[#3a3a3a] to-[#2a2a2a] rounded-lg p-1.5 border-2 border-blue-500 shadow-lg">
          {/* Master label */}
          <div className="text-[10px] font-bold text-blue-400 mb-1">
            MASTER
          </div>

          {/* Meter + Volume slider container - same size as tracks */}
          <div className="h-24 flex items-center gap-2 mb-0.5">
            {/* Master dB Meter - same size as track meters */}
            <div className="flex flex-col items-center gap-1">
              <div className="w-1.5 h-20 bg-gray-800 rounded-full overflow-hidden relative border border-gray-600">
                <div
                  className={`absolute bottom-0 w-full transition-all duration-75 ${
                    ((masterLevel + 60) / 60) * 100 > 90 ? 'bg-red-500' :
                    ((masterLevel + 60) / 60) * 100 > 70 ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`}
                  style={{ height: `${Math.max(0, Math.min(100, ((masterLevel + 60) / 60) * 100))}%` }}
                />
                {/* Peak indicators */}
                {[0, 25, 50, 75, 90].map(threshold => (
                  <div
                    key={threshold}
                    className="absolute w-full h-px bg-gray-500"
                    style={{ bottom: `${threshold}%` }}
                  />
                ))}
              </div>
              {/* dB readout */}
              <div className="text-[9px] font-mono text-green-400 w-6 text-center">
                {masterLevel === -Infinity || masterLevel <= -60 ? '-∞' : masterLevel.toFixed(0)}
              </div>
            </div>

            {/* Master Volume slider (rotated, same size as tracks) */}
            <input
              type="range"
              min="-60"
              max="12"
              step="1"
              value={masterVolume}
              onChange={(e) => handleLocalMasterVolumeChange(Number(e.target.value))}
              className="w-24 origin-center -rotate-90 appearance-none bg-gray-700 rounded-full h-1.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-md"
            />
          </div>

          {/* Master Volume display and Pan control - combined row */}
          <div className="flex items-center justify-between w-full mb-1 px-1">
            <div className="text-[9px] font-mono text-blue-300 font-bold">
              {masterVolume}dB
            </div>
            <div className="text-[8px] font-mono text-green-400">
              {masterPan === 0 ? 'C' :
               masterPan < 0 ? `L${Math.abs(masterPan)}` :
               `R${masterPan}`}
            </div>
          </div>

          {/* Master Pan control slider */}
          <div className="w-full mb-1 px-1">
            <input
              type="range"
              min="-100"
              max="100"
              step="1"
              value={masterPan}
              onChange={(e) => handleLocalMasterPanChange(Number(e.target.value))}
              className="w-full appearance-none bg-gray-700 rounded-full h-1 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-green-500 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>

          {/* Spacer with icon */}
          <div className="h-[22px] mb-0.5 flex items-center justify-center">
            <Music className="w-4 h-4 text-blue-400 opacity-50" />
          </div>

          {/* Spacer to match instrument name height */}
          <div className="text-[9px] text-gray-500 h-[13px]">
            &nbsp;
          </div>
        </div>
      </div>
    </div>
  );
});
