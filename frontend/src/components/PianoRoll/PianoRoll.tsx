"use client";

import { useState, useRef, useEffect, useCallback, JSX } from "react";
import { ParsedTrack } from "@/lib/dslParser";
import { SNAP_OPTIONS } from "@/components/Timeline/types";
import {
  PianoRollNote,
  MIDI_MIN,
  MIDI_MAX,
  NOTE_HEIGHT,
  PIANO_WIDTH,
  INSTRUMENT_GAINS,
  noteToPitch,
  pitchToNote,
  isBlackKey,
  getGridSubdivision,
  parseNotesFromDSL,
  updateDSLWithNewNotes,
  getInstrumentFromDSL,
  buildSamplerUrls,
  analyzeInstrumentGain,
  getTempoFromDSL,
  beatsToSeconds,
  secondsToBeats,
  snapToGrid,
  timeToX,
  xToTime,
  pitchToY,
  yToPitch,
} from "./pianoRollHelpers";
import { createNoteHandlers } from "./pianoRollHandlers";

interface PianoRollProps {
  track: ParsedTrack;
  dslCode: string;
  onCodeChange: (newCode: string) => void;
  isPlaying: boolean;
  currentTime: number;
  onCompile?: () => void;
  onPlay?: () => void;
  onStop?: () => void;
  onSolo?: () => void;
  isSoloed?: boolean;
  isLoading?: boolean;
  onSeek?: (time: number) => void;
}

export function PianoRoll({ track, dslCode, onCodeChange, isPlaying, currentTime, onCompile, onPlay, onStop, onSolo, isSoloed, isLoading, onSeek }: PianoRollProps) {
  // State
  const [zoom, setZoom] = useState(50);
  const [selectedNotes, setSelectedNotes] = useState<Set<number>>(new Set());
  const [draggingNote, setDraggingNote] = useState<{
    noteIndex: number;
    startX: number;
    startY: number;
    initialStart: number;
    initialPitch: number;
    currentStart?: number;
    currentPitch?: number;
  } | null>(null);
  const [resizingNote, setResizingNote] = useState<{
    noteIndex: number;
    chordIndices: number[]; // All notes in the chord (including noteIndex)
    startX: number;
    initialDuration: number;
    currentDuration?: number;
  } | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ time: number; pitch: number } | null>(null);
  const [isBoxSelecting, setIsBoxSelecting] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const [boxEnd, setBoxEnd] = useState<{ x: number; y: number } | null>(null);
  const [potentialBoxSelect, setPotentialBoxSelect] = useState<{ x: number; y: number } | null>(null);
  const [clipboard, setClipboard] = useState<PianoRollNote[]>([]);
  const [clipboardOriginTime, setClipboardOriginTime] = useState<number>(0);
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showPastePreview, setShowPastePreview] = useState(false);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [snapValue, setSnapValue] = useState(0.25);
  const [samplerLoaded, setSamplerLoaded] = useState(false);
  const [availableNotes, setAvailableNotes] = useState<Set<number>>(new Set());
  const [editingVelocity, setEditingVelocity] = useState<{
    noteIndex: number;
    startY: number;
    initialVelocity: number;
    currentVelocity?: number;
  } | null>(null);
  const [hoveredVelocityNote, setHoveredVelocityNote] = useState<number | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);
  const velocityCanvasRef = useRef<HTMLDivElement>(null);
  const samplerRef = useRef<any>(null);

  // Get tempo from DSL
  const tempo = getTempoFromDSL(dslCode);
  const dslInstrument = getInstrumentFromDSL(dslCode, track.id);

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

  // Helper wrappers that close over component state for cleaner usage
  const parseNotes = () => parseNotesFromDSL(dslCode, track.id, tempo);
  const updateNotes = (notes: PianoRollNote[]) =>
    updateDSLWithNewNotes(dslCode, track.id, notes, tempo, onCodeChange);
  const snapTo = (beats: number) => snapToGrid(beats, snapValue, snapEnabled);
  const beatToX = (beats: number) => timeToX(beats, zoom);
  const xToBeat = (x: number) => xToTime(x, zoom);

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

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH + canvasRef.current.scrollLeft;
    const y = e.clientY - rect.top + canvasRef.current.scrollTop;

    // Ignore clicks on scrollbar (typically last 17px of container)
    const clickYRelativeToContainer = e.clientY - rect.top;
    const containerHeight = canvasRef.current.clientHeight;
    const scrollbarHeight = 17; // Typical scrollbar height
    if (clickYRelativeToContainer > containerHeight - scrollbarHeight && canvasRef.current.scrollWidth > canvasRef.current.clientWidth) {
      return; // Clicked on horizontal scrollbar
    }

    if (x < 0) return;

    // Check if clicking on an existing note
    const notes = parseNotes();
    const clickedNoteIndex = notes.findIndex(note => {
      const noteX = beatToX(note.start);
      const noteY = pitchToY(note.pitch);
      const noteWidth = beatToX(note.duration);
      return x >= noteX && x <= noteX + noteWidth &&
        y >= noteY && y <= noteY + NOTE_HEIGHT;
    });

    if (clickedNoteIndex !== -1) {
      const clickedNote = notes[clickedNoteIndex];

      // Don't allow selection of loop-generated notes
      if (clickedNote.isFromLoop) {
        return;
      }

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
    const x = e.clientX - rect.left - PIANO_WIDTH + canvasRef.current.scrollLeft;
    const y = e.clientY - rect.top + canvasRef.current.scrollTop;

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
      const notes = parseNotes();
      const minX = Math.min(boxStart.x, x);
      const maxX = Math.max(boxStart.x, x);
      const minY = Math.min(boxStart.y, y);
      const maxY = Math.max(boxStart.y, y);

      const notesInBox = new Set<number>();
      notes.forEach((note, idx) => {
        // Skip loop-generated notes - they can't be selected
        if (note.isFromLoop) return;

        const noteX = beatToX(note.start);
        const noteY = pitchToY(note.pitch);
        const noteWidth = beatToX(note.duration);
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
      const currentTime = xToBeat(x);
      // Visual feedback during draw (could add a preview note here)
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - PIANO_WIDTH + canvasRef.current.scrollLeft;
    const y = e.clientY - rect.top + canvasRef.current.scrollTop;

    // Ignore clicks on scrollbar (typically last 17px of container)
    const clickYRelativeToContainer = e.clientY - rect.top;
    const containerHeight = canvasRef.current.clientHeight;
    const scrollbarHeight = 17; // Typical scrollbar height
    if (clickYRelativeToContainer > containerHeight - scrollbarHeight && canvasRef.current.scrollWidth > canvasRef.current.clientWidth) {
      // Clicked on horizontal scrollbar - clear any potential selections but don't create notes
      setPotentialBoxSelect(null);
      return;
    }

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
      const clickedBeats = xToBeat(potentialBoxSelect.x);
      const snappedBeats = snapTo(clickedBeats);
      const pitch = yToPitch(potentialBoxSelect.y);

      const notes = parseNotes();
      notes.push({
        pitch,
        start: snappedBeats,
        duration: snapValue,
        velocity: 0.8
      });

      updateNotes(notes);
      setPotentialBoxSelect(null);
      return;
    }

    // Clear potential box select
    setPotentialBoxSelect(null);

    // Note drawing
    if (isDrawing && drawStart) {
      const endBeats = xToBeat(x);

      // Calculate the raw drag distance
      const dragDistance = Math.abs(endBeats - drawStart.time);

      let duration: number;

      // If barely dragged (less than half a snap value), create note at snap size
      if (dragDistance < snapValue / 2) {
        duration = snapValue;
      } else {
        // User dragged - snap the end point and calculate duration
        const snappedEndBeats = snapTo(endBeats);
        duration = Math.max(snapValue, Math.abs(snappedEndBeats - drawStart.time));
      }

      const notes = parseNotes();
      notes.push({
        pitch: drawStart.pitch,
        start: drawStart.time,
        duration,
        velocity: 0.8
      });

      updateNotes(notes);
      setIsDrawing(false);
      setDrawStart(null);
    }
  };

  const handleNoteMouseDown = (e: React.MouseEvent, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    e.preventDefault(); // Prevent text selection
    const notes = parseNotes();
    const note = notes[noteIndex];

    // Prevent editing loop-generated notes
    if (note.isFromLoop) {
      return;
    }

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
      // If this is a chord note, find all notes at the same start time
      const noteToResize = notes[noteIndex];
      let chordIndices = [noteIndex];

      if (noteToResize.isChord) {
        const chordStartTime = noteToResize.start;
        chordIndices = notes
          .map((n, idx) => ({ note: n, idx }))
          .filter(({ note }) =>
            note.isChord &&
            Math.abs(note.start - chordStartTime) < 0.001 && // Same start time (with tolerance)
            !note.isFromLoop // Don't include loop-generated notes
          )
          .map(({ idx }) => idx);

        console.log(`[PianoRoll] Resizing chord with ${chordIndices.length} notes at time ${chordStartTime}`);
      }

      setResizingNote({
        noteIndex,
        chordIndices,
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
      e.preventDefault(); // Prevent text selection during drag

      const deltaX = e.clientX - draggingNote.startX;
      const deltaY = e.clientY - draggingNote.startY;
      const deltaBeats = deltaX / zoom;
      const deltaPitch = Math.round(-deltaY / NOTE_HEIGHT);

      const rawStart = draggingNote.initialStart + deltaBeats;
      const snappedStart = snapTo(Math.max(0, rawStart));
      const newPitch = Math.max(MIDI_MIN, Math.min(MIDI_MAX, draggingNote.initialPitch + deltaPitch));

      // Update visual state only
      setDraggingNote({
        ...draggingNote,
        currentStart: snappedStart,
        currentPitch: newPitch
      });
    } else if (resizingNote) {
      e.preventDefault(); // Prevent text selection during drag

      const deltaX = e.clientX - resizingNote.startX;
      const deltaBeats = deltaX / zoom;

      const rawDuration = resizingNote.initialDuration + deltaBeats;
      const snappedDuration = Math.max(snapValue, snapTo(rawDuration));

      // Update visual state only
      setResizingNote({
        ...resizingNote,
        currentDuration: snappedDuration
      });
    }
  };

  const handleMouseUp = () => {
    // Apply DSL changes if note was actually moved
    if (draggingNote && draggingNote.currentStart !== undefined && draggingNote.currentPitch !== undefined) {
      const notes = parseNotes();
      notes[draggingNote.noteIndex].start = draggingNote.currentStart;
      notes[draggingNote.noteIndex].pitch = draggingNote.currentPitch;
      updateNotes(notes);
    }

    if (resizingNote && resizingNote.currentDuration !== undefined) {
      const notes = parseNotes();

      // Update duration for all notes in the chord
      resizingNote.chordIndices.forEach(idx => {
        notes[idx].duration = resizingNote.currentDuration!;
      });

      console.log(`[PianoRoll] Updated duration for ${resizingNote.chordIndices.length} chord notes to ${resizingNote.currentDuration}`);
      updateNotes(notes);
    }

    setDraggingNote(null);
    setResizingNote(null);
  };

  const handleDeleteNote = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotes();
    // Filter out selected notes (iterate in reverse to avoid index issues)
    const filteredNotes = notes.filter((_, idx) => !selectedNotes.has(idx));
    updateNotes(filteredNotes);
    setSelectedNotes(new Set());
  };

  const handleCopy = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotes();
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

    const notes = parseNotes();

    // Paste at mouse position (snapped to grid)
    const pasteBeats = snapTo(xToBeat(mousePosition.x));

    // Calculate offset from original position
    const timeOffset = pasteBeats - clipboardOriginTime;

    // Create new notes with offset time
    const newNotes = clipboard.map(note => ({
      ...note,
      start: note.start + timeOffset
    }));

    // Add to existing notes
    notes.push(...newNotes);
    updateNotes(notes);

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
    const notes = parseNotes();

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

    updateNotes(notes);
  };

  const handleSelectAll = () => {
    const notes = parseNotes();
    const allIndices = new Set<number>();
    notes.forEach((_, idx) => allIndices.add(idx));
    setSelectedNotes(allIndices);
  };

  const handleDuplicate = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotes();
    const selectedNotesList = Array.from(selectedNotes).map(idx => notes[idx]);

    // Find the rightmost (latest) note
    const latestEnd = Math.max(...selectedNotesList.map(n => n.start + n.duration));

    // Offset for duplicates: place them right after the selection
    // Use snap value for clean offset
    const offset = snapTo(latestEnd) - Math.min(...selectedNotesList.map(n => n.start));

    // Create duplicates with offset
    const duplicates = selectedNotesList.map(note => ({
      ...note,
      start: note.start + offset
    }));

    // Add to existing notes
    notes.push(...duplicates);
    updateNotes(notes);

    // Select the newly duplicated notes
    const startIndex = notes.length - duplicates.length;
    const newSelection = new Set<number>();
    for (let i = 0; i < duplicates.length; i++) {
      newSelection.add(startIndex + i);
    }
    setSelectedNotes(newSelection);
  };

  const handleInvertSelection = () => {
    const notes = parseNotes();
    const newSelection = new Set<number>();

    notes.forEach((_, idx) => {
      if (!selectedNotes.has(idx)) {
        newSelection.add(idx);
      }
    });

    setSelectedNotes(newSelection);
  };

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

  const handleVelocityMouseDown = (e: React.MouseEvent, noteIndex: number) => {
    e.stopPropagation();
    e.preventDefault(); // Prevent text selection
    const notes = parseNotes();
    const note = notes[noteIndex];

    // Prevent editing velocity of loop-generated notes
    if (note.isFromLoop) {
      return;
    }

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

  const handleVelocityMouseMove = useCallback((e: MouseEvent) => {
    setEditingVelocity(prev => {
      if (!prev) return null;

      e.preventDefault(); // Prevent text selection during drag

      const deltaY = prev.startY - e.clientY; // Inverted: up = increase
      const velocityChange = deltaY / 120; // 120px = 1.0 velocity change (matches visual bar scale)

      const newVelocity = Math.max(0.1, Math.min(1.0, prev.initialVelocity + velocityChange));

      // Update visual state only (don't update DSL during drag)
      return {
        ...prev,
        currentVelocity: newVelocity
      };
    });
  }, []);

  const handleVelocityMouseUp = useCallback(() => {
    setEditingVelocity(prev => {
      // Apply DSL changes if velocity was actually changed
      if (prev && prev.currentVelocity !== undefined) {
        const notes = parseNotesFromDSL(dslCode, track.id, tempo);

        // Find all notes at the same time position (they're part of the same chord)
        const editedNote = notes[prev.noteIndex];
        const startTime = editedNote.start;

        // Update velocity for ALL notes at this time position (entire chord)
        notes.forEach((note, idx) => {
          if (Math.abs(note.start - startTime) < 0.001) {
            notes[idx].velocity = prev.currentVelocity;
          }
        });

        updateDSLWithNewNotes(dslCode, track.id, notes, tempo, onCodeChange);
      }

      return null;
    });
  }, [dslCode, track.id, tempo, onCodeChange]);

  useEffect(() => {
    if (editingVelocity) {
      window.addEventListener('mousemove', handleVelocityMouseMove);
      window.addEventListener('mouseup', handleVelocityMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleVelocityMouseMove);
        window.removeEventListener('mouseup', handleVelocityMouseUp);
      };
    }
  }, [editingVelocity, handleVelocityMouseMove, handleVelocityMouseUp]);

  // Add global event listeners for note dragging and resizing
  useEffect(() => {
    if (draggingNote || resizingNote) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [draggingNote, resizingNote, handleMouseMove, handleMouseUp]);

  // Sync horizontal scroll between piano roll and velocity editor
  useEffect(() => {
    const pianoRoll = canvasRef.current;
    const velocityEditor = velocityCanvasRef.current;
    if (!pianoRoll || !velocityEditor) return;

    const handlePianoRollScroll = () => {
      if (velocityEditor.scrollLeft !== pianoRoll.scrollLeft) {
        velocityEditor.scrollLeft = pianoRoll.scrollLeft;
      }
    };

    const handleVelocityEditorScroll = () => {
      if (pianoRoll.scrollLeft !== velocityEditor.scrollLeft) {
        pianoRoll.scrollLeft = velocityEditor.scrollLeft;
      }
    };

    pianoRoll.addEventListener('scroll', handlePianoRollScroll);
    velocityEditor.addEventListener('scroll', handleVelocityEditorScroll);

    return () => {
      pianoRoll.removeEventListener('scroll', handlePianoRollScroll);
      velocityEditor.removeEventListener('scroll', handleVelocityEditorScroll);
    };
  }, []);

  // Auto-scroll to keep playhead visible (like DAWs)
  useEffect(() => {
    if (!canvasRef.current || !isPlaying) return;

    const container = canvasRef.current;
    const playheadPosition = secondsToBeats(currentTime, tempo) * zoom + PIANO_WIDTH;
    const viewportLeft = container.scrollLeft;
    const viewportRight = viewportLeft + container.clientWidth;
    const scrollMargin = 100; // Keep playhead this many pixels from edge

    // Check if playhead is going off the right edge
    if (playheadPosition > viewportRight - scrollMargin) {
      // Scroll to keep playhead at first 1/4 of viewport (more content ahead visible)
      container.scrollLeft = playheadPosition - container.clientWidth / 4 - PIANO_WIDTH;
    }
    // Check if playhead is going off the left edge (when looping back)
    else if (playheadPosition < viewportLeft + scrollMargin + PIANO_WIDTH) {
      container.scrollLeft = Math.max(0, playheadPosition - container.clientWidth / 4 - PIANO_WIDTH);
    }
  }, [currentTime, isPlaying, zoom, tempo]);

  const notes = parseNotes();
  const maxDuration = Math.max(10, ...notes.map(n => n.start + n.duration));

  return (
    <div className="bg-gray-950 border border-white/10 rounded-xl overflow-hidden flex flex-col h-full min-h-0">
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

          {/* Playback controls */}
          <div className="flex items-center gap-2">
            {onCompile && (
              <button
                onClick={onCompile}
                className="px-3 py-1 text-xs font-semibold rounded-lg transition-colors bg-green-600 hover:bg-green-500 text-white"
                title="Compile & Play Track"
              >
                COMPILE
              </button>
            )}
            {(onPlay || onStop) && (
              <button
                onClick={isPlaying ? onStop : onPlay}
                className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${
                  isPlaying
                    ? 'bg-red-600 hover:bg-red-500'
                    : 'bg-blue-600 hover:bg-blue-500'
                } text-white`}
                title={isPlaying ? "Stop Playback" : "Play Track"}
              >
                {isPlaying ? 'STOP' : 'PLAY'}
              </button>
            )}
            {onSolo && (
              <button
                onClick={onSolo}
                className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${
                  isSoloed
                    ? 'bg-yellow-600 text-white'
                    : 'bg-white/10 text-gray-400 hover:bg-white/20'
                }`}
                title={isSoloed ? "Unsolo Track" : "Solo Track"}
              >
                {isSoloed ? 'SOLOED' : 'SOLO'}
              </button>
            )}
          </div>

          {/* Snap controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSnapEnabled(!snapEnabled)}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${snapEnabled
                  ? 'bg-blue-600 text-white'
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

      {/* Playhead Scrubber Header */}
      {onSeek && (
        <div className="bg-gray-900 border-b border-white/10 px-4 py-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 min-w-[60px]">
              {Math.floor(currentTime / 60)}:{String(Math.floor(currentTime % 60)).padStart(2, '0')}.
              {String(Math.floor((currentTime % 1) * 10)).padStart(1, '0')}
            </span>
            <input
              type="range"
              min="0"
              max={beatsToSeconds(maxDuration, tempo)}
              step="0.01"
              value={currentTime}
              onChange={(e) => onSeek(parseFloat(e.target.value))}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentTime / beatsToSeconds(maxDuration, tempo)) * 100}%, #374151 ${(currentTime / beatsToSeconds(maxDuration, tempo)) * 100}%, #374151 100%)`
              }}
            />
            <span className="text-xs text-gray-400 min-w-[60px] text-right">
              {Math.floor(beatsToSeconds(maxDuration, tempo) / 60)}:{String(Math.floor(beatsToSeconds(maxDuration, tempo) % 60)).padStart(2, '0')}
            </span>
          </div>
        </div>
      )}

      {/* Loading Modal */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 rounded-xl">
          <div className="bg-gray-800 rounded-lg p-6 text-center shadow-2xl border border-gray-700">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
            <div className="text-white font-semibold">Loading Audio...</div>
            <div className="text-gray-400 text-sm mt-1">Please wait</div>
          </div>
        </div>
      )}

      <div
        ref={canvasRef}
        className="relative overflow-auto flex-1 min-h-0 select-none"
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
                  className={`absolute border-b ${!hasSample
                      ? 'bg-gray-950/50 border-gray-800/30'
                      : isBlack
                        ? 'bg-gray-900 border-gray-800'
                        : 'bg-gray-950 border-gray-700'
                    }`}
                  style={{
                    top: `${i * NOTE_HEIGHT}px`,
                    height: `${NOTE_HEIGHT}px`,
                    width: `${maxDuration * zoom}px`,
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
              const isFromLoop = note.isFromLoop;

              // Check if this note is being dragged or resized
              const isDragging = draggingNote?.noteIndex === idx;
              const isResizing = resizingNote?.chordIndices?.includes(idx) ?? false;

              // Use current position if dragging, otherwise use note position
              const displayStart = (isDragging && draggingNote.currentStart !== undefined)
                ? draggingNote.currentStart
                : note.start;
              const displayPitch = (isDragging && draggingNote.currentPitch !== undefined)
                ? draggingNote.currentPitch
                : note.pitch;
              const displayDuration = (isResizing && resizingNote?.currentDuration !== undefined)
                ? resizingNote.currentDuration
                : note.duration;

              return (
                <div
                  key={idx}
                  className={`absolute rounded ${isFromLoop ? 'cursor-not-allowed' : 'cursor-move'} ${
                    !hasSample
                      ? 'bg-red-500/50 border-2 border-red-600'
                      : isFromLoop
                        ? 'bg-gray-600 border border-gray-500'
                        : note.isChord
                          ? 'bg-blue-500'
                          : 'bg-blue-600'
                  } ${isSelected ? 'ring-2 ring-white' : ''} ${!isFromLoop && 'hover:brightness-110'}`}
                  style={{
                    left: `${beatToX(displayStart)}px`,
                    top: `${pitchToY(displayPitch) + 1}px`,
                    width: `${beatToX(displayDuration)}px`,
                    height: `${NOTE_HEIGHT - 2}px`,
                    opacity: hasSample ? (isFromLoop ? 0.6 : note.velocity) : 0.5,
                  }}
                  onMouseDown={(e) => hasSample && !isFromLoop && handleNoteMouseDown(e, idx, false)}
                  title={isFromLoop ? 'Loop-generated note (read-only)' : undefined}
                >
                  <div className="text-xs text-white px-1 truncate pointer-events-none flex items-center gap-1">
                    <span>{pitchToNote(displayPitch)}</span>
                    {!hasSample && <span>⚠</span>}
                    {isFromLoop && <span className="text-[10px]">🔒</span>}
                  </div>

                  {hasSample && !isFromLoop && (
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
                  left: `${beatToX(snapTo(xToBeat(mousePosition.x)))}px`,
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
                style={{ left: `${beatToX(secondsToBeats(currentTime, tempo))}px` }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Velocity Editor */}
      <div className="flex-none border-t border-white/10">
        <div className="bg-gray-900 px-4 py-2 border-b border-white/10">
          <h4 className="text-sm font-semibold text-white">Velocity Editor</h4>
          <p className="text-xs text-gray-400">Click and drag bars to adjust velocity</p>
        </div>
        <div
          ref={velocityCanvasRef}
          className="relative overflow-x-auto overflow-y-hidden bg-gray-950 select-none"
          style={{ height: '140px' }}
        >
          <div className="flex">
            {/* Spacer for piano keys alignment */}
            <div className="sticky left-0 z-10 bg-gray-900" style={{ width: `${PIANO_WIDTH}px` }}>
              <div className="h-full flex items-center justify-center text-xs text-gray-500">
                Velocity
              </div>
            </div>

            {/* Velocity bars */}
            <div className="relative flex-1" style={{ width: `${maxDuration * zoom}px`, height: '140px', paddingTop: '20px' }}>
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
                const isFromLoop = note.isFromLoop;

                // Check if this note is being edited
                const isEditingThis = editingVelocity?.noteIndex === idx;

                // Use current velocity if editing, otherwise use note velocity
                const displayVelocity = (isEditingThis && editingVelocity.currentVelocity !== undefined)
                  ? editingVelocity.currentVelocity
                  : note.velocity;

                const barHeight = displayVelocity * 120; // 0-1 → 0-120px
                const percentage = Math.round(displayVelocity * 100);

                return (
                  <div
                    key={idx}
                    className={`absolute ${isFromLoop ? 'cursor-not-allowed' : 'cursor-ns-resize'} ${
                      isFromLoop
                        ? 'bg-gradient-to-t from-gray-600 via-gray-500 to-gray-400'
                        : isSelected
                          ? 'bg-gradient-to-t from-blue-500 via-blue-400 to-blue-300'
                          : 'bg-gradient-to-t from-blue-700 via-blue-600 to-blue-500'
                      } ${!isFromLoop && 'hover:brightness-110'} transition-all shadow-lg`}
                    style={{
                      left: `${beatToX(note.start)}px`,
                      bottom: '0px',
                      width: `${beatToX(note.duration)}px`,
                      height: `${barHeight}px`,
                      opacity: isFromLoop ? 0.6 : 1,
                    }}
                    onMouseDown={(e) => !isFromLoop && handleVelocityMouseDown(e, idx)}
                    onMouseEnter={() => !isFromLoop && setHoveredVelocityNote(idx)}
                    onMouseLeave={() => setHoveredVelocityNote(null)}
                    title={isFromLoop ? `Loop note - Velocity: ${percentage}% (read-only)` : `Velocity: ${percentage}%`}
                  >
                    {(isSelected || isHovered) && !isFromLoop && (
                      <div className="absolute -top-5 left-0 right-0 text-center text-xs text-white font-bold bg-blue-900/80 rounded px-1">
                        {percentage}%
                      </div>
                    )}
                    {isFromLoop && (isHovered || isSelected) && (
                      <div className="absolute -top-5 left-0 right-0 text-center text-xs text-white font-bold bg-gray-700/80 rounded px-1">
                        🔒 {percentage}%
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