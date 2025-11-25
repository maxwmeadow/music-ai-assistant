"use client";

import { useState, useRef, useEffect, JSX } from "react";
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
const MIDI_MAX = 108; // C8 (full 88-key piano range)
const NOTE_HEIGHT = 12; // pixels per semitone
const PIANO_WIDTH = 60; // width of piano keys

// Pre-calculated gain compensation map (in dB)
// Target level: -12dB peak
const INSTRUMENT_GAINS: Record<string, number> = {
  // Pianos
  'piano/steinway_grand': 0,
  'piano/bechstein_1911_upright': 0,
  'piano/fender_rhodes': 0,
  'piano/experience_ny_steinway': 0,
  // Harpsichords
  'harpsichord/harpsichord_english': -10,
  'harpsichord/harpsichord_flemish': -10,
  'harpsichord/harpsichord_french': -10,
  'harpsichord/harpsichord_italian': -15,
  'harpsichord/harpsichord_unk': -10,
  // Guitars
  'guitar/rjs_guitar_palm_muted_softly_strings': -5,
  'guitar/rjs_guitar_palm_muted_strings': -6,
  'synth/lead/ld_the_stack_guitar_chug': 0,
  'synth/lead/ld_the_stack_guitar': -1,
  'guitar/rjs_guitar_new_strings': 0,
  'guitar/rjs_guitar_old_strings': 4,
  // Bass
  'bass/funky_fingers': -10,
  'bass/low_fat_bass': -5,
  'bass/jp8000_sawbass': 2,
  'bass/jp8000_tribass': 2,
  // Strings
  'strings/nfo_chamber_strings_longs': 0,
  'strings/nfo_iso_celli_swells': 0,
  'strings/nfo_iso_viola_swells': 0,
  'strings/nfo_iso_violin_swells': 0,
  // Brass
  'brass/nfo_iso_brass_swells': 0,
  // Winds
  'winds/flute_violin': 0,
  'winds/subtle_clarinet': 0,
  'winds/decent_oboe': 0,
  'winds/tenor_saxophone': 0,
};


export function PianoRoll({ track, dslCode, onCodeChange, isPlaying, currentTime }: PianoRollProps) {
  const [zoom, setZoom] = useState(50); // pixels per second
  const [selectedNotes, setSelectedNotes] = useState<Set<number>>(new Set());
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

  // Box selection state
  const [isBoxSelecting, setIsBoxSelecting] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const [boxEnd, setBoxEnd] = useState<{ x: number; y: number } | null>(null);
  const [potentialBoxSelect, setPotentialBoxSelect] = useState<{ x: number; y: number } | null>(null);

  // Clipboard state
  const [clipboard, setClipboard] = useState<PianoRollNote[]>([]);
  const [clipboardOriginTime, setClipboardOriginTime] = useState<number>(0);
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showPastePreview, setShowPastePreview] = useState(false);

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

  // Parse instrument from DSL code directly (not from compiled track)
  const getInstrumentFromDSL = (): string | null => {
    const trackMatch = dslCode.match(new RegExp(`track\\s*\\(\\s*["']${track.id}["']\\s*\\)\\s*\\{([\\s\\S]*?)\\}`, 'm'));
    if (!trackMatch) return null;
    const trackContent = trackMatch[1];
    const instrumentMatch = trackContent.match(/instrument\s*\(\s*["']([^"']+)["']\s*\)/);
    return instrumentMatch ? instrumentMatch[1] : null;
  };

  const dslInstrument = getInstrumentFromDSL();

  // Load instrument sampler for preview
  useEffect(() => {
    loadInstrumentPreview();
    return () => {
      if (samplerRef.current) {
        samplerRef.current.dispose();
      }
    };
  }, [dslInstrument]);

  const loadInstrumentPreview = async () => {
    const instrumentToLoad = dslInstrument || track.instrument;
    if (!instrumentToLoad) return;

    try {
      const Tone = await import('tone');

      const CDN_BASE = 'https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev';

      // Fetch instrument mapping
      const mappingUrl = `${CDN_BASE}/samples/${instrumentToLoad}/mapping.json`;
      const response = await fetch(mappingUrl, { cache: 'force-cache' });

      if (!response.ok) {
        console.warn('Could not load instrument mapping for preview');
        return;
      }

      const mapping = await response.json();
      const { urls, baseUrl } = buildSamplerUrls(mapping, instrumentToLoad, CDN_BASE);

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

      // Use pre-calculated gain if available, otherwise analyze dynamically
      let calculatedGain: number;
      if (instrumentToLoad && INSTRUMENT_GAINS.hasOwnProperty(instrumentToLoad)) {
        calculatedGain = INSTRUMENT_GAINS[instrumentToLoad];
        console.log('[PianoRoll] Using pre-calculated gain for', instrumentToLoad + ':', calculatedGain + 'dB');
      } else {
        console.log('[PianoRoll] Analyzing gain for', instrumentToLoad);
        calculatedGain = await analyzeInstrumentGain(urls, baseUrl);
        console.log('[PianoRoll] Applying', calculatedGain.toFixed(1), 'dB gain');
      }

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

      console.log('[PianoRoll] Loaded instrument preview:', instrumentToLoad);
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

    return { urls, baseUrl };
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
    { label: '1/4 (Whole)', value: 4 },
    { label: '1/2 (Half)', value: 2 },
    { label: '1 (Quarter)', value: 1 },
    { label: '1/2 (8th)', value: 0.5 },
    { label: '1/4 (16th)', value: 0.25 },
    { label: '1/8 (32nd)', value: 0.125 },
  ];

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH;
    const y = e.clientY - rect.top;

    if (x < 0) return;

    // Check if clicking on an existing note
    const notes = parseNotesFromDSL(track.id);
    const clickedNoteIndex = notes.findIndex(note => {
      const noteX = timeToX(note.start);
      const noteY = pitchToY(note.pitch);
      const noteWidth = timeToX(note.duration);
      return x >= noteX && x <= noteX + noteWidth &&
        y >= noteY && y <= noteY + NOTE_HEIGHT;
    });

    if (clickedNoteIndex !== -1) {
      // Clicked on a note - handle selection
      if (e.shiftKey) {
        // Shift+click: toggle note in selection
        setSelectedNotes(prev => {
          const newSet = new Set(prev);
          if (newSet.has(clickedNoteIndex)) {
            newSet.delete(clickedNoteIndex);
          } else {
            newSet.add(clickedNoteIndex);
          }
          return newSet;
        });
      } else {
        // Regular click: select only this note
        setSelectedNotes(new Set([clickedNoteIndex]));
      }
      return;
    }

    // Clicked on empty space
    if (!e.shiftKey) {
      // Clear selection if not holding shift
      setSelectedNotes(new Set());
    }

    // Mark potential box selection - will only activate if user drags
    setPotentialBoxSelect({ x, y });
  };


  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH;
    const y = e.clientY - rect.top;

    // Track mouse position for paste-at-cursor functionality
    setMousePosition({ x, y });

    // Check if we should activate box selection from potential click
    if (potentialBoxSelect && !isBoxSelecting) {
      const dragDistance = Math.sqrt(
        Math.pow(x - potentialBoxSelect.x, 2) + Math.pow(y - potentialBoxSelect.y, 2)
      );

      // Only activate box selection if user has dragged at least 5 pixels
      if (dragDistance > 5) {
        setIsBoxSelecting(true);
        setBoxStart(potentialBoxSelect);
        setBoxEnd({ x, y });
        setPotentialBoxSelect(null);
      }
      return;
    }

    // Box selection
    if (isBoxSelecting && boxStart) {
      setBoxEnd({ x, y });

      // Calculate which notes are in the selection box
      const notes = parseNotesFromDSL(track.id);
      const minX = Math.min(boxStart.x, x);
      const maxX = Math.max(boxStart.x, x);
      const minY = Math.min(boxStart.y, y);
      const maxY = Math.max(boxStart.y, y);

      const notesInBox = new Set<number>();
      notes.forEach((note, idx) => {
        const noteX = timeToX(note.start);
        const noteY = pitchToY(note.pitch);
        const noteWidth = timeToX(note.duration);
        const noteHeight = NOTE_HEIGHT;

        // Check if note overlaps with selection box
        const noteRight = noteX + noteWidth;
        const noteBottom = noteY + noteHeight;

        if (noteX < maxX && noteRight > minX &&
          noteY < maxY && noteBottom > minY) {
          notesInBox.add(idx);
        }
      });

      setSelectedNotes(notesInBox);
    }

    // Note drawing feedback
    if (isDrawing && drawStart) {
      const currentTime = xToTime(x);
      // Visual feedback during draw (could add a preview note here)
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH;
    const y = e.clientY - rect.top;

    // End box selection
    if (isBoxSelecting) {
      setIsBoxSelecting(false);
      setBoxStart(null);
      setBoxEnd(null);
      return;
    }

    // If there was a potential box select that never activated (single click on empty space)
    // Create a new note at that position
    if (potentialBoxSelect && x >= 0) {
      const clickedBeats = xToTime(potentialBoxSelect.x);
      const snappedBeats = snapToGrid(clickedBeats);
      const pitch = yToPitch(potentialBoxSelect.y);

      const notes = parseNotesFromDSL(track.id);
      notes.push({
        pitch,
        start: snappedBeats,
        duration: snapValue,
        velocity: 0.8
      });

      updateDSLWithNewNotes(notes);
      setPotentialBoxSelect(null);
      return;
    }

    // Clear potential box select
    setPotentialBoxSelect(null);

    // Note drawing
    if (isDrawing && drawStart) {
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
    }
  };

  const handleNoteMouseDown = (e: React.MouseEvent, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    const notes = parseNotesFromDSL(track.id);

    // If shift-clicking, toggle selection
    if (e.shiftKey) {
      setSelectedNotes(prev => {
        const newSet = new Set(prev);
        if (newSet.has(noteIndex)) {
          newSet.delete(noteIndex);
        } else {
          newSet.add(noteIndex);
        }
        return newSet;
      });
      return;
    }

    // If clicking on a note that's already selected, keep the selection
    // Otherwise, select only this note
    if (!selectedNotes.has(noteIndex)) {
      setSelectedNotes(new Set([noteIndex]));
    }

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
    if (selectedNotes.size === 0) return;

    const notes = parseNotesFromDSL(track.id);
    // Filter out selected notes (iterate in reverse to avoid index issues)
    const filteredNotes = notes.filter((_, idx) => !selectedNotes.has(idx));
    updateDSLWithNewNotes(filteredNotes);
    setSelectedNotes(new Set());
  };

  const handleCopy = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotesFromDSL(track.id);
    const selectedNotesList = Array.from(selectedNotes).map(idx => notes[idx]);

    // Find earliest start time to use as origin
    const earliestStart = Math.min(...selectedNotesList.map(n => n.start));

    setClipboard(selectedNotesList);
    setClipboardOriginTime(earliestStart);
  };

  const handleCut = () => {
    handleCopy();
    handleDeleteNote();
  };

  const handlePaste = () => {
    if (clipboard.length === 0) return;

    const notes = parseNotesFromDSL(track.id);

    // Paste at mouse position (snapped to grid)
    const pasteBeats = snapToGrid(xToTime(mousePosition.x));

    // Calculate offset from original position
    const timeOffset = pasteBeats - clipboardOriginTime;

    // Create new notes with offset time
    const newNotes = clipboard.map(note => ({
      ...note,
      start: note.start + timeOffset
    }));

    // Add to existing notes
    notes.push(...newNotes);
    updateDSLWithNewNotes(notes);

    // Select the newly pasted notes
    const startIndex = notes.length - newNotes.length;
    const newSelection = new Set<number>();
    for (let i = 0; i < newNotes.length; i++) {
      newSelection.add(startIndex + i);
    }
    setSelectedNotes(newSelection);

    // Show brief visual feedback
    setShowPastePreview(true);
    setTimeout(() => setShowPastePreview(false), 300);
  };

  const handleQuantize = (gridValue: number) => {
    const notes = parseNotesFromDSL(track.id);

    // Quantize selected notes (or all if none selected)
    const indicesToQuantize = selectedNotes.size > 0
      ? Array.from(selectedNotes)
      : notes.map((_, idx) => idx);

    indicesToQuantize.forEach(idx => {
      const note = notes[idx];
      // Snap start time to grid
      note.start = Math.round(note.start / gridValue) * gridValue;
      // Snap duration to grid (with minimum duration)
      note.duration = Math.max(gridValue, Math.round(note.duration / gridValue) * gridValue);
    });

    updateDSLWithNewNotes(notes);
  };

  const handleSelectAll = () => {
    const notes = parseNotesFromDSL(track.id);
    const allIndices = new Set<number>();
    notes.forEach((_, idx) => allIndices.add(idx));
    setSelectedNotes(allIndices);
  };

  const handleDuplicate = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotesFromDSL(track.id);
    const selectedNotesList = Array.from(selectedNotes).map(idx => notes[idx]);

    // Find the rightmost (latest) note
    const latestEnd = Math.max(...selectedNotesList.map(n => n.start + n.duration));

    // Offset for duplicates: place them right after the selection
    // Use snap value for clean offset
    const offset = snapToGrid(latestEnd) - Math.min(...selectedNotesList.map(n => n.start));

    // Create duplicates with offset
    const duplicates = selectedNotesList.map(note => ({
      ...note,
      start: note.start + offset
    }));

    // Add to existing notes
    notes.push(...duplicates);
    updateDSLWithNewNotes(notes);

    // Select the newly duplicated notes
    const startIndex = notes.length - duplicates.length;
    const newSelection = new Set<number>();
    for (let i = 0; i < duplicates.length; i++) {
      newSelection.add(startIndex + i);
    }
    setSelectedNotes(newSelection);
  };

  const handleInvertSelection = () => {
    const notes = parseNotesFromDSL(track.id);
    const newSelection = new Set<number>();

    notes.forEach((_, idx) => {
      if (!selectedNotes.has(idx)) {
        newSelection.add(idx);
      }
    });

    setSelectedNotes(newSelection);
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
      // Check if we're in an input (don't trigger if typing in snap dropdown, etc.)
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA') {
        return;
      }

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;

      // Delete
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        handleDeleteNote();
      }

      // Copy: Ctrl+C / Cmd+C
      if (ctrlOrCmd && e.key === 'c') {
        e.preventDefault();
        handleCopy();
      }

      // Cut: Ctrl+X / Cmd+X
      if (ctrlOrCmd && e.key === 'x') {
        e.preventDefault();
        handleCut();
      }

      // Paste: Ctrl+V / Cmd+V
      if (ctrlOrCmd && e.key === 'v') {
        e.preventDefault();
        handlePaste();
      }

      // Select All: Ctrl+A / Cmd+A
      if (ctrlOrCmd && e.key === 'a') {
        e.preventDefault();
        handleSelectAll();
      }

      // Duplicate: Ctrl+D / Cmd+D
      if (ctrlOrCmd && e.key === 'd') {
        e.preventDefault();
        handleDuplicate();
      }

      // Deselect all: Escape
      if (e.key === 'Escape') {
        e.preventDefault();
        setSelectedNotes(new Set());
      }

      // Invert selection: Ctrl+I / Cmd+I
      if (ctrlOrCmd && e.key === 'i') {
        e.preventDefault();
        handleInvertSelection();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNotes, clipboard, clipboardOriginTime, isPlaying, currentTime]);

  // Velocity editor state
  const [editingVelocity, setEditingVelocity] = useState<{
    noteIndex: number;
    startY: number;
    initialVelocity: number;
  } | null>(null);
  const [hoveredVelocityNote, setHoveredVelocityNote] = useState<number | null>(null);
  const velocityCanvasRef = useRef<HTMLDivElement>(null);

  const handleVelocityMouseDown = (e: React.MouseEvent, noteIndex: number) => {
    e.stopPropagation();
    const notes = parseNotesFromDSL(track.id);
    setEditingVelocity({
      noteIndex,
      startY: e.clientY,
      initialVelocity: notes[noteIndex].velocity
    });

    // Select this note if not already selected
    if (!selectedNotes.has(noteIndex)) {
      setSelectedNotes(new Set([noteIndex]));
    }
  };

  const handleVelocityMouseMove = (e: MouseEvent) => {
    if (!editingVelocity) return;

    const deltaY = editingVelocity.startY - e.clientY; // Inverted: up = increase
    const velocityChange = deltaY / 100; // 100px = 1.0 velocity change

    const notes = parseNotesFromDSL(track.id);
    const newVelocity = Math.max(0.1, Math.min(1.0, editingVelocity.initialVelocity + velocityChange));

    notes[editingVelocity.noteIndex].velocity = newVelocity;
    updateDSLWithNewNotes(notes);
  };

  const handleVelocityMouseUp = () => {
    setEditingVelocity(null);
  };

  useEffect(() => {
    if (editingVelocity) {
      window.addEventListener('mousemove', handleVelocityMouseMove);
      window.addEventListener('mouseup', handleVelocityMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleVelocityMouseMove);
        window.removeEventListener('mouseup', handleVelocityMouseUp);
      };
    }
  }, [editingVelocity]);

  const notes = parseNotesFromDSL(track.id);
  const maxDuration = Math.max(10, ...notes.map(n => n.start + n.duration));

  return (
    <div className="bg-gray-950 border border-white/10 rounded-xl overflow-hidden flex flex-col">
      <div className="bg-gray-900 border-b border-white/10 p-4 flex items-center justify-between">
        <div>
          <h3 className="text-white font-semibold">Piano Roll - {track.id}</h3>
          <p className="text-xs text-gray-400">
            Click piano keys to preview • Click and drag grid to create notes • Drag notes to move • Delete key to
            remove
          </p>
          {selectedNotes.size > 0 && (
            <p className="text-xs text-blue-400 font-semibold mt-1">
              {selectedNotes.size} note{selectedNotes.size > 1 ? 's' : ''} selected
            </p>
          )}
        </div>
        <div className="flex items-center gap-4">
          {!samplerLoaded && track.instrument && (
            <span className="text-xs text-yellow-400">Loading samples...</span>
          )}

          {/* Snap controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSnapEnabled(!snapEnabled)}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${snapEnabled
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

          {/* Quantize controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleQuantize(snapValue)}
              disabled={notes.length === 0}
              className="px-3 py-1 text-xs font-semibold rounded-lg transition-colors bg-blue-600 hover:bg-blue-500 disabled:bg-white/10 disabled:text-gray-500 text-white"
              title={selectedNotes.size > 0 ? `Quantize ${selectedNotes.size} selected notes` : 'Quantize all notes'}
            >
              QUANTIZE
            </button>
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
        style={{ height: '400px' }}
        onMouseDown={handleCanvasClick}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
      >
        <div className="flex">
          {/* Piano keys */}
          <div className="sticky left-0 z-10 bg-gray-900" style={{ width: `${PIANO_WIDTH}px` }}>
            {Array.from({ length: MIDI_MAX - MIDI_MIN + 1 }).map((_, i) => {
              const pitch = MIDI_MAX - i;
              const isBlack = isBlackKey(pitch);
              const isC = pitch % 12 === 0;
              const hasSample = availableNotes.has(pitch);

              return (
                <div
                  key={pitch}
                  className={`border-b border-gray-700 ${!hasSample
                      ? 'bg-gray-950 cursor-not-allowed'
                      : isBlack
                        ? 'bg-gray-800 hover:bg-gray-700 cursor-pointer'
                        : 'bg-gray-100 hover:bg-gray-200 cursor-pointer'
                    } transition-colors`}
                  style={{ height: `${NOTE_HEIGHT}px` }}
                  onClick={() => hasSample && handlePianoKeyClick(pitch)}
                >
                  <span className={`text-xs px-2 ${!hasSample
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
          <div className="relative flex-1" style={{ width: `${maxDuration * zoom}px` }}>
            {/* Grid lines */}
            {Array.from({ length: MIDI_MAX - MIDI_MIN + 1 }).map((_, i) => {
              const pitch = MIDI_MAX - i;
              const isBlack = isBlackKey(pitch);
              const hasSample = availableNotes.has(pitch);

              return (
                <div
                  key={pitch}
                  className={`absolute w-full border-b ${!hasSample
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
              const isSelected = selectedNotes.has(idx);

              return (
                <div
                  key={idx}
                  className={`absolute rounded cursor-move ${!hasSample
                      ? 'bg-red-500/50 border-2 border-red-600'
                      : note.isChord
                        ? 'bg-blue-500'
                        : 'bg-purple-500'
                    } ${isSelected ? 'ring-2 ring-white' : ''} hover:brightness-110`}
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

            {/* Box selection visual */}
            {isBoxSelecting && boxStart && boxEnd && (
              <div
                className="absolute border-2 border-blue-400 bg-blue-400/20 pointer-events-none"
                style={{
                  left: `${Math.min(boxStart.x, boxEnd.x)}px`,
                  top: `${Math.min(boxStart.y, boxEnd.y)}px`,
                  width: `${Math.abs(boxEnd.x - boxStart.x)}px`,
                  height: `${Math.abs(boxEnd.y - boxStart.y)}px`,
                }}
              />
            )}

            {/* Paste preview animation */}
            {showPastePreview && (
              <div
                className="absolute border-2 border-green-400 bg-green-400/30 pointer-events-none animate-pulse rounded"
                style={{
                  left: `${timeToX(snapToGrid(xToTime(mousePosition.x)))}px`,
                  top: `${mousePosition.y - 20}px`,
                  width: '60px',
                  height: '40px',
                }}
              >
                <div className="text-green-100 text-xs font-bold text-center pt-2">
                  Pasted!
                </div>
              </div>
            )}

            {/* Playback cursor */}
            {isPlaying && (
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20 pointer-events-none"
                style={{ left: `${timeToX(secondsToBeats(currentTime))}px` }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Velocity Editor */}
      <div className="border-t border-white/10">
        <div className="bg-gray-900 px-4 py-2 border-b border-white/10">
          <h4 className="text-sm font-semibold text-white">Velocity Editor</h4>
          <p className="text-xs text-gray-400">Click and drag bars to adjust velocity</p>
        </div>
        <div
          ref={velocityCanvasRef}
          className="relative overflow-auto bg-gray-950"
          style={{ height: '120px' }}
        >
          <div className="flex">
            {/* Spacer for piano keys alignment */}
            <div className="sticky left-0 z-10 bg-gray-900" style={{ width: `${PIANO_WIDTH}px` }}>
              <div className="h-full flex items-center justify-center text-xs text-gray-500">
                Velocity
              </div>
            </div>

            {/* Velocity bars */}
            <div className="relative flex-1" style={{ width: `${maxDuration * zoom}px`, height: '100px' }}>
              {/* Background grid lines (same as piano roll time markers) */}
              {Array.from({ length: Math.ceil(maxDuration / 4) + 1 }).map((_, i) => (
                <div
                  key={i}
                  className="absolute top-0 bottom-0 border-l-2 border-gray-600"
                  style={{ left: `${i * 4 * zoom}px` }}
                />
              ))}

              {/* Velocity bars for each note */}
              {notes.map((note, idx) => {
                const isSelected = selectedNotes.has(idx);
                const isHovered = hoveredVelocityNote === idx;
                const barHeight = note.velocity * 100; // 0-1 → 0-100px
                const percentage = Math.round(note.velocity * 100);

                return (
                  <div
                    key={idx}
                    className={`absolute cursor-ns-resize ${
                      isSelected
                        ? 'bg-gradient-to-t from-purple-500 via-purple-400 to-purple-300'
                        : 'bg-gradient-to-t from-purple-700 via-purple-600 to-purple-500'
                      } hover:brightness-110 transition-all shadow-lg`}
                    style={{
                      left: `${timeToX(note.start)}px`,
                      bottom: '0px',
                      width: `${timeToX(note.duration)}px`,
                      height: `${barHeight}px`,
                    }}
                    onMouseDown={(e) => handleVelocityMouseDown(e, idx)}
                    onMouseEnter={() => setHoveredVelocityNote(idx)}
                    onMouseLeave={() => setHoveredVelocityNote(null)}
                    title={`Velocity: ${percentage}%`}
                  >
                    {(isSelected || isHovered) && (
                      <div className="absolute -top-5 left-0 right-0 text-center text-xs text-white font-bold bg-purple-900/80 rounded px-1">
                        {percentage}%
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}