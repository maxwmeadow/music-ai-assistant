"use client";

import {useState, useRef, useEffect, JSX} from "react";
import { ParsedTrack } from "@/lib/dslParser";

interface PianoRollNote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  isChord?: boolean;
}

interface PianoRollProps {
  track: ParsedTrack;
  dslCode: string;
  onCodeChange: (newCode: string) => void;
  isPlaying: boolean;
  currentTime: number;
}

const MIDI_MIN = 36; // C2
const MIDI_MAX = 84; // C6
const NOTE_HEIGHT = 12; // pixels per semitone
const PIANO_WIDTH = 60; // width of piano keys


export function PianoRoll({ track, dslCode, onCodeChange, isPlaying, currentTime }: PianoRollProps) {
  const [zoom, setZoom] = useState(50); // pixels per second
  const [selectedNote, setSelectedNote] = useState<number | null>(null);
  const [draggingNote, setDraggingNote] = useState<{
    noteIndex: number;
    startX: number;
    startY: number;
    initialStart: number;
    initialPitch: number;
  } | null>(null);
  const [resizingNote, setResizingNote] = useState<{
    noteIndex: number;
    startX: number;
    initialDuration: number;
  } | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ time: number; pitch: number } | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);

  const [snapEnabled, setSnapEnabled] = useState(true);
  const [snapValue, setSnapValue] = useState(0.25);

  const getTempoFromDSL = (): number => {
    const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
    return tempoMatch ? parseInt(tempoMatch[1]) : 120;
  };

  const beatsToSeconds = (beats: number): number => {
    const tempo = getTempoFromDSL();
    return (beats * 60) / tempo;
  };

  const secondsToBeats = (seconds: number): number => {
    const tempo = getTempoFromDSL();
    return (seconds * tempo) / 60;
  };

  const snapToGrid = (beats: number): number => {
    if (!snapEnabled) return beats;
    return Math.round(beats / snapValue) * snapValue;
  };

  /**
   * Determines grid subdivision level based on zoom
   * Returns: { subdivision: number, showSubdivisions: boolean }
   */
  const getGridSubdivision = (zoom: number): { subdivision: number, showSubdivisions: boolean, showSixteenths: boolean } => {
    // zoom is pixels per beat

    if (zoom < 60) {
      // Very zoomed out - only show beats (quarters)
      return { subdivision: 1, showSubdivisions: false, showSixteenths: false };
    } else if (zoom < 120) {
      // Medium zoom - show eighths
      return { subdivision: 0.5, showSubdivisions: true, showSixteenths: false };
    } else {
      // Very zoomed in - show sixteenths
      return { subdivision: 0.25, showSubdivisions: true, showSixteenths: true };
    }
  };

  const parseNotesFromDSL = (trackId: string): PianoRollNote[] => {
    const trackMatch = dslCode.match(new RegExp(`track\\("${trackId}"\\)\\s*{([^}]+)}`, 's'));
    if (!trackMatch) return [];

    const trackContent = trackMatch[1];
    const notes: PianoRollNote[] = [];

    const noteMatches = trackContent.matchAll(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
    for (const match of noteMatches) {
      const [, noteName, start, duration, velocity] = match;
      const pitch = noteToPitch(noteName);

      notes.push({
        pitch,
        start: secondsToBeats(parseFloat(start)), // Convert to beats
        duration: secondsToBeats(parseFloat(duration)), // Convert to beats
        velocity: parseFloat(velocity)
      });
    }

    const chordMatches = trackContent.matchAll(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
    for (const match of chordMatches) {
      const [, notesStr, start, duration, velocity] = match;
      const chordNotes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));

      chordNotes.forEach(noteName => {
        const pitch = noteToPitch(noteName);
        notes.push({
          pitch,
          start: secondsToBeats(parseFloat(start)), // Convert to beats
          duration: secondsToBeats(parseFloat(duration)), // Convert to beats
          velocity: parseFloat(velocity),
          isChord: true
        });
      });
    }

    return notes;
  };

  const noteToPitch = (noteName: string): number => {
    const noteMap: Record<string, number> = {
      'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
      'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    };
    const match = noteName.match(/([A-G]#?)(\d+)/);
    if (!match) return 60;
    const [, note, octave] = match;
    return (parseInt(octave) + 1) * 12 + noteMap[note];
  };

  const pitchToNote = (pitch: number): string => {
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(pitch / 12) - 1;
    const note = notes[pitch % 12];
    return `${note}${octave}`;
  };

  const isBlackKey = (pitch: number): boolean => {
    const semitone = pitch % 12;
    return [1, 3, 6, 8, 10].includes(semitone); // C#, D#, F#, G#, A#
  };

  const updateDSLWithNewNotes = (updatedNotes: PianoRollNote[]) => {
    const trackMatch = dslCode.match(new RegExp(`(track\\("${track.id}"\\)\\s*{)([^}]+)(})`, 's'));
    if (!trackMatch) return;

    const [fullMatch, opening, , closing] = trackMatch;
    const instrumentMatch = trackMatch[2].match(/instrument\("([^"]+)"\)/);
    const instrumentLine = instrumentMatch ? `  instrument("${instrumentMatch[1]}")\n` : '';

    const notesByTime = new Map<string, PianoRollNote[]>();
    updatedNotes.forEach(note => {
      const timeKey = note.start.toFixed(3);
      if (!notesByTime.has(timeKey)) {
        notesByTime.set(timeKey, []);
      }
      notesByTime.get(timeKey)!.push(note);
    });

    const noteLines: string[] = [];
    Array.from(notesByTime.entries())
        .sort(([a], [b]) => parseFloat(a) - parseFloat(b))
        .forEach(([timeStr, notes]) => {
          if (notes.length === 1) {
            const note = notes[0];
            const noteName = pitchToNote(note.pitch);
            // Convert beats back to seconds for DSL
            const startSeconds = beatsToSeconds(note.start);
            const durationSeconds = beatsToSeconds(note.duration);
            noteLines.push(
                `  note("${noteName}", ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(1)})`
            );
          } else {
            const noteNames = notes.map(n => `"${pitchToNote(n.pitch)}"`).join(', ');
            const avgDuration = notes.reduce((sum, n) => sum + n.duration, 0) / notes.length;
            const avgVelocity = notes.reduce((sum, n) => sum + n.velocity, 0) / notes.length;
            // Convert beats back to seconds for DSL
            const startSeconds = beatsToSeconds(notes[0].start);
            const durationSeconds = beatsToSeconds(avgDuration);
            noteLines.push(
                `  chord([${noteNames}], ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${avgVelocity.toFixed(1)})`
            );
          }
        });

    const newTrackContent = `${opening}\n${instrumentLine}${noteLines.join('\n')}\n${closing}`;
    const newDSL = dslCode.replace(fullMatch, newTrackContent);
    onCodeChange(newDSL);
  };

  const [samplerLoaded, setSamplerLoaded] = useState(false);
  const [availableNotes, setAvailableNotes] = useState<Set<number>>(new Set());
  const samplerRef = useRef<any>(null);

  // Load instrument sampler for preview
  useEffect(() => {
    loadInstrumentPreview();
    return () => {
      if (samplerRef.current) {
        samplerRef.current.dispose();
      }
    };
  }, [track.instrument]);

  const loadInstrumentPreview = async () => {
    if (!track.instrument) return;

    try {
      const Tone = await import('tone');

      const CDN_BASE = 'https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev';

      // Fetch instrument mapping
      const mappingUrl = `${CDN_BASE}/samples/${track.instrument}/mapping.json`;
      const response = await fetch(mappingUrl, {cache: 'force-cache'});

      if (!response.ok) {
        console.warn('Could not load instrument mapping for preview');
        return;
      }

      const mapping = await response.json();
      const {urls, baseUrl} = buildSamplerUrls(mapping, track.instrument, CDN_BASE);

      // Track which MIDI notes have samples
      const available = new Set<number>();

      if (mapping.velocity_layers) {
        Object.values(mapping.velocity_layers).forEach((layers: any) => {
          const layer = layers[0];
          if (layer.lokey !== undefined && layer.hikey !== undefined) {
            for (let note = layer.lokey; note <= layer.hikey; note++) {
              if (note >= MIDI_MIN && note <= MIDI_MAX) {
                available.add(note);
              }
            }
          }
        });
      } else if (mapping.samples) {
        Object.keys(urls).forEach(noteName => {
          const pitch = noteToPitch(noteName);
          if (pitch >= MIDI_MIN && pitch <= MIDI_MAX) {
            available.add(pitch);
          }
        });
      }

      setAvailableNotes(available);

      // Analyze gain
      console.log('[PianoRoll] Analyzing gain for', track.instrument);
      const calculatedGain = await analyzeInstrumentGain(urls, baseUrl);
      console.log('[PianoRoll] Applying', calculatedGain.toFixed(1), 'dB gain');

      // Dispose old sampler if exists
      if (samplerRef.current) {
        samplerRef.current.dispose();
      }

      // Create new sampler with calculated gain
      samplerRef.current = await new Promise((resolve, reject) => {
        const sampler = new Tone.Sampler({
          urls,
          baseUrl,
          volume: calculatedGain, // Use calculated gain instead of hardcoded -10
          onload: () => {
            setSamplerLoaded(true);
            resolve(sampler);
          },
          onerror: reject
        }).toDestination();
      });

      console.log('[PianoRoll] Loaded instrument preview:', track.instrument);
    } catch (error) {
      console.error('[PianoRoll] Failed to load instrument:', error);
    }
  };

  // Add the gain analysis function (same as in server.js)
  const analyzeInstrumentGain = async (urls: Record<string, string>, baseUrl: string): Promise<number> => {
    try {
      const Tone = await import('tone');
      const sampleKeys = Object.keys(urls).slice(0, Math.min(3, Object.keys(urls).length));
      if (sampleKeys.length === 0) return 0;

      const audioContext = Tone.context.rawContext;
      const peakLevels: number[] = [];

      for (const key of sampleKeys) {
        try {
          const sampleUrl = baseUrl + urls[key];
          const response = await fetch(sampleUrl);
          const arrayBuffer = await response.arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          let maxPeak = 0;
          for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let i = 0; i < channelData.length; i++) {
              maxPeak = Math.max(maxPeak, Math.abs(channelData[i]));
            }
          }

          peakLevels.push(maxPeak);
          console.log(`[PianoRoll Gain] Sample ${key}: peak = ${maxPeak.toFixed(6)} (${(20 * Math.log10(maxPeak)).toFixed(1)}dB)`);
        } catch (err) {
          console.warn(`[PianoRoll Gain] Failed to analyze ${key}:`, err);
        }
      }

      if (peakLevels.length === 0) return 0;

      const avgPeak = peakLevels.reduce((sum, p) => sum + p, 0) / peakLevels.length;
      const avgPeakDb = 20 * Math.log10(avgPeak);
      const targetDb = -6;
      const neededGain = targetDb - avgPeakDb;

      console.log(`[PianoRoll Gain] Average peak: ${avgPeak.toFixed(6)} (${avgPeakDb.toFixed(1)}dB)`);
      console.log(`[PianoRoll Gain] Recommended gain: ${neededGain.toFixed(1)}dB`);

      return Math.max(-20, Math.min(60, neededGain));
    } catch (error) {
      console.error('[PianoRoll Gain] Error:', error);
      return 0;
    }
  };

  const buildSamplerUrls = (mapping: any, instrumentPath: string, cdnBase: string) => {
    const urls: Record<string, string> = {};
    const baseUrl = `${cdnBase}/samples/${instrumentPath}/`;

    if (mapping.velocity_layers) {
      for (const [note, layers] of Object.entries(mapping.velocity_layers)) {
        const layerArray = layers as any[];

        // Get the lokey/hikey range from the first layer to understand coverage
        const sampleLayer = layerArray.find((l: any) =>
            l.file.includes('Sustains') || l.file.includes('sus')
        ) || layerArray[0];

        if (!sampleLayer) continue;

        // Get the pitch_center - this is what note the sample actually is
        const pitchCenter = sampleLayer.pitch_center;

        if (pitchCenter !== undefined) {
          const noteName = pitchToNote(pitchCenter);
          urls[noteName] = sampleLayer.file.split('/').map(encodeURIComponent).join('/');
        }
      }
    } else if (mapping.samples) {
      for (const [note, file] of Object.entries(mapping.samples)) {
        urls[note] = (file as string).split('/').map(encodeURIComponent).join('/');
      }
    }

    return {urls, baseUrl};
  };

  const playPreviewNote = async (pitch: number) => {
    if (!samplerRef.current || !samplerLoaded || !availableNotes.has(pitch)) return;

    try {
      const Tone = await import('tone');
      if (Tone.context.state !== 'running') {
        await Tone.start();
      }

      const noteName = pitchToNote(pitch);
      samplerRef.current.triggerAttackRelease(noteName, '8n');
    } catch (error) {
      console.error('[PianoRoll] Error playing preview:', error);
    }
  };

  const handlePianoKeyClick = (pitch: number) => {
    playPreviewNote(pitch);
  };

  const timeToX = (beats: number) => beats * zoom;
  const xToTime = (x: number) => x / zoom;
  const pitchToY = (pitch: number) => (MIDI_MAX - pitch) * NOTE_HEIGHT;
  const yToPitch = (y: number) => Math.round(MIDI_MAX - y / NOTE_HEIGHT);

  const SNAP_OPTIONS = [
    {label: '1/4 (Whole)', value: 4},
    {label: '1/2 (Half)', value: 2},
    {label: '1 (Quarter)', value: 1},
    {label: '1/2 (8th)', value: 0.5},
    {label: '1/4 (16th)', value: 0.25},
    {label: '1/8 (32nd)', value: 0.125},
  ];

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH;
    const y = e.clientY - rect.top;

    if (x < 0) return;

    const rawBeats = xToTime(x);
    const snappedBeats = snapToGrid(rawBeats);
    const pitch = yToPitch(y);

    if (pitch < MIDI_MIN || pitch > MIDI_MAX) return;

    setIsDrawing(true);
    setDrawStart({time: snappedBeats, pitch});
  };


  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !drawStart || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH;
    const currentTime = xToTime(x);

    // Visual feedback during draw (could add a preview note here)
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (!isDrawing || !drawStart || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH;
    const endBeats = xToTime(x);

    // Calculate the raw drag distance
    const dragDistance = Math.abs(endBeats - drawStart.time);

    console.log('=== Note Creation Debug ===');
    console.log('snapValue:', snapValue);
    console.log('drawStart.time:', drawStart.time);
    console.log('endBeats:', endBeats);
    console.log('dragDistance:', dragDistance);
    console.log('dragDistance < snapValue / 2?', dragDistance < snapValue / 2);

    let duration: number;

    // If barely dragged (less than half a snap value), create note at snap size
    if (dragDistance < snapValue / 2) {
      duration = snapValue;
      console.log('Using snap duration:', duration);
    } else {
      // User dragged - snap the end point and calculate duration
      const snappedEndBeats = snapToGrid(endBeats);
      duration = Math.max(snapValue, Math.abs(snappedEndBeats - drawStart.time));
      console.log('Using dragged duration:', duration);
    }

    const notes = parseNotesFromDSL(track.id);
    notes.push({
      pitch: drawStart.pitch,
      start: drawStart.time,
      duration,
      velocity: 0.8
    });

    updateDSLWithNewNotes(notes);
    setIsDrawing(false);
    setDrawStart(null);
  };

  const handleNoteMouseDown = (e: React.MouseEvent, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    const notes = parseNotesFromDSL(track.id);

    if (isResize) {
      setResizingNote({
        noteIndex,
        startX: e.clientX,
        initialDuration: notes[noteIndex].duration
      });
    } else {
      setDraggingNote({
        noteIndex,
        startX: e.clientX,
        startY: e.clientY,
        initialStart: notes[noteIndex].start,
        initialPitch: notes[noteIndex].pitch
      });
    }
    setSelectedNote(noteIndex);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (draggingNote) {
      const deltaX = e.clientX - draggingNote.startX;
      const deltaY = e.clientY - draggingNote.startY;
      const deltaBeats = deltaX / zoom;
      const deltaPitch = Math.round(-deltaY / NOTE_HEIGHT);

      const notes = parseNotesFromDSL(track.id);
      const rawStart = draggingNote.initialStart + deltaBeats;
      const snappedStart = snapToGrid(Math.max(0, rawStart));

      notes[draggingNote.noteIndex].start = snappedStart;
      notes[draggingNote.noteIndex].pitch = Math.max(
          MIDI_MIN,
          Math.min(MIDI_MAX, draggingNote.initialPitch + deltaPitch)
      );

      updateDSLWithNewNotes(notes);
    } else if (resizingNote) {
      const deltaX = e.clientX - resizingNote.startX;
      const deltaBeats = deltaX / zoom;

      const notes = parseNotesFromDSL(track.id);
      const rawDuration = resizingNote.initialDuration + deltaBeats;
      const snappedDuration = Math.max(snapValue, snapToGrid(rawDuration));

      notes[resizingNote.noteIndex].duration = snappedDuration;

      updateDSLWithNewNotes(notes);
    }
  };

  const handleMouseUp = () => {
    setDraggingNote(null);
    setResizingNote(null);
  };

  const handleDeleteNote = () => {
    if (selectedNote === null) return;

    const notes = parseNotesFromDSL(track.id);
    notes.splice(selectedNote, 1);
    updateDSLWithNewNotes(notes);
    setSelectedNote(null);
  };

  useEffect(() => {
    if (draggingNote || resizingNote) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [draggingNote, resizingNote]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        handleDeleteNote();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNote]);

  const notes = parseNotesFromDSL(track.id);
  const maxDuration = Math.max(10, ...notes.map(n => n.start + n.duration));

  return (
      <div className="bg-gray-950 border border-white/10 rounded-xl overflow-hidden">
        <div className="bg-gray-900 border-b border-white/10 p-4 flex items-center justify-between">
          <div>
            <h3 className="text-white font-semibold">Piano Roll - {track.id}</h3>
            <p className="text-xs text-gray-400">
              Click piano keys to preview • Click and drag grid to create notes • Drag notes to move • Delete key to
              remove
            </p>
          </div>
          <div className="flex items-center gap-4">
            {!samplerLoaded && track.instrument && (
                <span className="text-xs text-yellow-400">Loading samples...</span>
            )}

            {/* Snap controls */}
            <div className="flex items-center gap-2">
              <button
                  onClick={() => setSnapEnabled(!snapEnabled)}
                  className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${
                      snapEnabled
                          ? 'bg-purple-600 text-white'
                          : 'bg-white/10 text-gray-400'
                  }`}
              >
                SNAP
              </button>

              <select
                  value={snapValue}
                  onChange={(e) => setSnapValue(Number(e.target.value))}
                  disabled={!snapEnabled}
                  className="bg-white/10 text-white text-xs rounded-lg px-2 py-1 border border-white/20 disabled:opacity-50"
              >
                {SNAP_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value} className="bg-gray-900">
                      {opt.label}
                    </option>
                ))}
              </select>
            </div>

            <label className="text-sm text-gray-400">Zoom:</label>
            <input
                type="range"
                min="20"
                max="200"
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
                className="w-32"
            />
          </div>
        </div>

        <div
            ref={canvasRef}
            className="relative overflow-auto"
            style={{height: '400px'}}
            onMouseDown={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
        >
          <div className="flex">
            {/* Piano keys */}
            <div className="sticky left-0 z-10 bg-gray-900" style={{width: `${PIANO_WIDTH}px`}}>
              {Array.from({length: MIDI_MAX - MIDI_MIN + 1}).map((_, i) => {
                const pitch = MIDI_MAX - i;
                const isBlack = isBlackKey(pitch);
                const isC = pitch % 12 === 0;
                const hasSample = availableNotes.has(pitch);

                return (
                    <div
                        key={pitch}
                        className={`border-b border-gray-700 ${
                            !hasSample
                                ? 'bg-gray-950 cursor-not-allowed'
                                : isBlack
                                    ? 'bg-gray-800 hover:bg-gray-700 cursor-pointer'
                                    : 'bg-gray-100 hover:bg-gray-200 cursor-pointer'
                        } transition-colors`}
                        style={{height: `${NOTE_HEIGHT}px`}}
                        onClick={() => hasSample && handlePianoKeyClick(pitch)}
                    >
                  <span className={`text-xs px-2 ${
                      !hasSample
                          ? 'text-gray-700'
                          : isBlack
                              ? 'text-gray-400'
                              : 'text-gray-600'
                  }`}>
                    {isC && pitchToNote(pitch)}
                    {!hasSample && ' ×'}
                  </span>
                    </div>
                );
              })}
            </div>

            {/* Grid and notes */}
            <div className="relative flex-1" style={{width: `${maxDuration * zoom}px`}}>
              {/* Grid lines */}
              {Array.from({length: MIDI_MAX - MIDI_MIN + 1}).map((_, i) => {
                const pitch = MIDI_MAX - i;
                const isBlack = isBlackKey(pitch);
                const hasSample = availableNotes.has(pitch);

                return (
                    <div
                        key={pitch}
                        className={`absolute w-full border-b ${
                            !hasSample
                                ? 'bg-gray-950/50 border-gray-800/30'
                                : isBlack
                                    ? 'bg-gray-900 border-gray-800'
                                    : 'bg-gray-950 border-gray-700'
                        }`}
                        style={{
                          top: `${i * NOTE_HEIGHT}px`,
                          height: `${NOTE_HEIGHT}px`,
                        }}
                    />
                );
              })}

              {/* Time markers with adaptive subdivisions */}
              {(() => {
                const { subdivision, showSubdivisions, showSixteenths } = getGridSubdivision(zoom);
                const totalBeats = Math.ceil(maxDuration);
                const markers: JSX.Element[] = [];

                // Generate all markers based on subdivision
                for (let beat = 0; beat <= totalBeats; beat += subdivision) {
                  const bar = Math.floor(beat / 4) + 1;
                  const beatInBar = (beat % 4);
                  const isMeasureStart = beat % 4 === 0;
                  const isBeatStart = beat % 1 === 0;
                  const isEighthNote = beat % 0.5 === 0 && beat % 1 !== 0;
                  const isSixteenthNote = beat % 0.25 === 0 && beat % 0.5 !== 0;

                  // Determine line style
                  let borderClass = '';
                  let labelContent = null;

                  if (isMeasureStart) {
                    borderClass = 'border-l-2 border-gray-500';
                    labelContent = (
                      <span className="text-xs font-bold text-gray-200 absolute -top-5 -translate-x-1/2">
                        {bar}
                      </span>
                    );
                  } else if (isBeatStart) {
                    borderClass = 'border-l border-gray-600';
                    labelContent = (
                      <span className="text-xs text-gray-400 absolute -top-5 -translate-x-1/2">
                        {Math.floor(beatInBar) + 1}
                      </span>
                    );
                  } else if (isEighthNote && showSubdivisions) {
                    borderClass = 'border-l border-gray-700/50';
                    if (showSixteenths) {
                      labelContent = (
                        <span className="text-[10px] text-gray-500 absolute -top-5 -translate-x-1/2">
                          +
                        </span>
                      );
                    }
                  } else if (isSixteenthNote && showSixteenths) {
                    borderClass = 'border-l border-gray-800/30';
                  }

                  markers.push(
                    <div
                      key={beat}
                      className={`absolute top-0 bottom-0 ${borderClass}`}
                      style={{ left: `${beat * zoom}px` }}
                    >
                      {labelContent}
                    </div>
                  );
                }

                return markers;
              })()}

              {/* Notes */}
              {notes.map((note, idx) => {
                const hasSample = availableNotes.has(note.pitch);

                return (
                    <div
                        key={idx}
                        className={`absolute rounded cursor-move ${
                            !hasSample
                                ? 'bg-red-500/50 border-2 border-red-600'
                                : note.isChord
                                    ? 'bg-blue-500'
                                    : 'bg-purple-500'
                        } ${selectedNote === idx ? 'ring-2 ring-white' : ''} hover:brightness-110`}
                        style={{
                          left: `${timeToX(note.start)}px`,
                          top: `${pitchToY(note.pitch) + 1}px`,
                          width: `${timeToX(note.duration)}px`,
                          height: `${NOTE_HEIGHT - 2}px`,
                          opacity: hasSample ? note.velocity : 0.5,
                        }}
                        onMouseDown={(e) => hasSample && handleNoteMouseDown(e, idx, false)}
                    >
                      <div className="text-xs text-white px-1 truncate pointer-events-none">
                        {pitchToNote(note.pitch)} {!hasSample && '⚠'}
                      </div>

                      {hasSample && (
                          <div
                              className="absolute right-0 top-0 bottom-0 w-1 bg-white/50 hover:bg-white cursor-ew-resize"
                              onMouseDown={(e) => handleNoteMouseDown(e, idx, true)}
                          />
                      )}
                    </div>
                );
              })}

              {/* Playback cursor */}
              {isPlaying && (
                  <div
                      className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20 pointer-events-none"
                      style={{left: `${timeToX(secondsToBeats(currentTime))}px`}}
                  />
              )}
            </div>
          </div>
        </div>
      </div>
  );
}