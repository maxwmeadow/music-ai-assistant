"use client";

import { useState, useRef, useEffect, JSX } from "react";
import { ParsedTrack } from "@/lib/dslParser";

interface TimelineNote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  isChord?: boolean;
}

interface TimelineProps {
  tracks: ParsedTrack[];
  dslCode: string;
  onCodeChange: (newCode: string) => void;
  isPlaying: boolean;
  currentTime: number;
  onSeek?: (time: number) => void;
  isLoading?: boolean;
}

export function Timeline({ tracks, dslCode, onCodeChange, isPlaying, currentTime, onSeek, isLoading = false }: TimelineProps) {
  const [zoom, setZoom] = useState(50);
  const [selectedNote, setSelectedNote] = useState<{ trackId: string, noteIndex: number } | null>(null);
  const [draggingNote, setDraggingNote] = useState<{ trackId: string, noteIndex: number, startX: number, initialStart: number } | null>(null);
  const [resizingNote, setResizingNote] = useState<{ trackId: string, noteIndex: number, startX: number, initialDuration: number } | null>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [snapValue, setSnapValue] = useState(0.25);
  const [copiedNote, setCopiedNote] = useState<{ note: TimelineNote, trackId: string } | null>(null);

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
   * Determines grid subdivision level based on snap value (when enabled) or zoom
   * Returns: { subdivision: number }
   */
  const getGridSubdivision = (zoom: number): { subdivision: number } => {
    // If snap is enabled, use the snap value to determine grid lines
    if (snapEnabled) {
      // Use exactly the snap value for grid spacing
      return { subdivision: snapValue };
    }

    // Otherwise, use zoom-based grid (original behavior)
    if (zoom < 60) {
      // Very zoomed out - only show beats (quarters)
      return { subdivision: 1 };
    } else if (zoom < 120) {
      // Medium zoom - show eighths
      return { subdivision: 0.5 };
    } else {
      // Very zoomed in - show sixteenths
      return { subdivision: 0.25 };
    }
  };

  const SNAP_OPTIONS = [
    { label: '1/4 (Whole)', value: 4 },
    { label: '1/2 (Half)', value: 2 },
    { label: '1 (Quarter)', value: 1 },
    { label: '1/2 (8th)', value: 0.5 },
    { label: '1/4 (16th)', value: 0.25 },
    { label: '1/8 (32nd)', value: 0.125 },
  ];

  const parseNotesFromDSL = (trackId: string): TimelineNote[] => {
    const trackMatch = dslCode.match(new RegExp(`track\\("${trackId}"\\)\\s*{([^}]+)}`, 's'));
    if (!trackMatch) return [];

    const trackContent = trackMatch[1];
    const notes: TimelineNote[] = [];

    // Parse regular notes
    const noteMatches = trackContent.matchAll(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
    for (const match of noteMatches) {
      const [, noteName, start, duration, velocity] = match;
      const pitch = noteToPitch(noteName);

      notes.push({
        pitch,
        start: secondsToBeats(parseFloat(start)),  // Convert to beats
        duration: secondsToBeats(parseFloat(duration)),  // Convert to beats
        velocity: parseFloat(velocity)
      });
    }

    // Parse chords
    const chordMatches = trackContent.matchAll(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
    for (const match of chordMatches) {
      const [, notesStr, start, duration, velocity] = match;
      const chordNotes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));
      const rootPitch = noteToPitch(chordNotes[0]);

      notes.push({
        pitch: rootPitch,
        start: secondsToBeats(parseFloat(start)),  // Convert to beats
        duration: secondsToBeats(parseFloat(duration)),  // Convert to beats
        velocity: parseFloat(velocity),
        isChord: true
      });
    }

    return notes.sort((a, b) => a.start - b.start);
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

  const updateDSLWithNewNotes = (trackId: string, updatedNotes: TimelineNote[]) => {
    const trackMatch = dslCode.match(new RegExp(`(track\\("${trackId}"\\)\\s*{)([^}]+)(})`, 's'));
    if (!trackMatch) return;

    const [fullMatch, opening, , closing] = trackMatch;

    const instrumentMatch = trackMatch[2].match(/instrument\("([^"]+)"\)/);
    const instrumentLine = instrumentMatch ? `  instrument("${instrumentMatch[1]}")\n` : '';

    // Generate new note lines, converting beats back to seconds
    const noteLines = updatedNotes.map(note => {
      const noteName = pitchToNote(note.pitch);
      const startSeconds = beatsToSeconds(note.start);
      const durationSeconds = beatsToSeconds(note.duration);

      if (note.isChord) {
        return `  chord(["${noteName}"], ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(1)})`;
      }
      return `  note("${noteName}", ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(1)})`;
    }).join('\n');

    const newTrackContent = `${opening}\n${instrumentLine}${noteLines}\n${closing}`;
    const newDSL = dslCode.replace(fullMatch, newTrackContent);

    onCodeChange(newDSL);
  };


  const handleNoteMouseDown = (e: React.MouseEvent, trackId: string, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    const notes = parseNotesFromDSL(trackId);

    if (isResize) {
      setResizingNote({
        trackId,
        noteIndex,
        startX: e.clientX,
        initialDuration: notes[noteIndex].duration
      });
    } else {
      setDraggingNote({
        trackId,
        noteIndex,
        startX: e.clientX,
        initialStart: notes[noteIndex].start
      });
    }
    setSelectedNote({ trackId, noteIndex });
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (draggingNote) {
      const deltaX = e.clientX - draggingNote.startX;
      const deltaBeats = deltaX / zoom;
      const rawStart = draggingNote.initialStart + deltaBeats;
      const newStart = snapToGrid(Math.max(0, rawStart));

      const notes = parseNotesFromDSL(draggingNote.trackId);
      notes[draggingNote.noteIndex].start = newStart;

      updateDSLWithNewNotes(draggingNote.trackId, notes);
    } else if (resizingNote) {
      const deltaX = e.clientX - resizingNote.startX;
      const deltaBeats = deltaX / zoom;
      const rawDuration = resizingNote.initialDuration + deltaBeats;
      const newDuration = Math.max(snapValue, snapToGrid(rawDuration));

      const notes = parseNotesFromDSL(resizingNote.trackId);
      notes[resizingNote.noteIndex].duration = newDuration;

      updateDSLWithNewNotes(resizingNote.trackId, notes);
    }
  };

  const handleMouseUp = () => {
    setDraggingNote(null);
    setResizingNote(null);
  };

  const handleDeleteNote = () => {
    if (!selectedNote) return;

    const notes = parseNotesFromDSL(selectedNote.trackId);
    notes.splice(selectedNote.noteIndex, 1);
    updateDSLWithNewNotes(selectedNote.trackId, notes);
    setSelectedNote(null);
  };

  const handleCopyNote = () => {
    if (!selectedNote) return;

    const notes = parseNotesFromDSL(selectedNote.trackId);
    const noteToCopy = notes[selectedNote.noteIndex];

    if (noteToCopy) {
      // Create a deep copy of the note
      setCopiedNote({
        note: { ...noteToCopy },
        trackId: selectedNote.trackId
      });
    }
  };

  const handlePasteNote = () => {
    if (!copiedNote) return;

    const { note, trackId } = copiedNote;
    const notes = parseNotesFromDSL(trackId);

    // Calculate the paste position at the playhead (red line)
    const pastePosition = secondsToBeats(currentTime);
    const snappedPosition = snapToGrid(pastePosition);

    // Create a new note at the playhead position
    const newNote: TimelineNote = {
      ...note,
      start: snappedPosition
    };

    // Add the new note and sort by start time
    notes.push(newNote);
    notes.sort((a, b) => a.start - b.start);

    updateDSLWithNewNotes(trackId, notes);

    // Select the newly pasted note
    const newNoteIndex = notes.findIndex(n => n.start === snappedPosition && n.pitch === newNote.pitch);
    if (newNoteIndex !== -1) {
      setSelectedNote({ trackId, noteIndex: newNoteIndex });
    }
  };

  const handleTimelineClick = (e: React.MouseEvent) => {
    // Don't allow interactions while loading
    if (isLoading) return;
    // Don't interfere with note dragging/resizing
    if (draggingNote || resizingNote) return;
    if (!onSeek || !timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left + timelineRef.current.scrollLeft;
    const clickedBeats = clickX / zoom;
    const clickedTime = beatsToSeconds(clickedBeats);

    onSeek(Math.max(0, clickedTime));
    setIsDraggingPlayhead(true);

    // Deselect any selected note when clicking blank area
    setSelectedNote(null);
  };

  const handlePlayheadDrag = (e: MouseEvent) => {
    if (!isDraggingPlayhead || !onSeek || !timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const dragX = e.clientX - rect.left + timelineRef.current.scrollLeft;
    const draggedBeats = dragX / zoom;
    const draggedTime = beatsToSeconds(draggedBeats);

    onSeek(Math.max(0, draggedTime));
  };

  const handlePlayheadDragEnd = () => {
    setIsDraggingPlayhead(false);
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
    if (isDraggingPlayhead) {
      window.addEventListener('mousemove', handlePlayheadDrag);
      window.addEventListener('mouseup', handlePlayheadDragEnd);
      return () => {
        window.removeEventListener('mousemove', handlePlayheadDrag);
        window.removeEventListener('mouseup', handlePlayheadDragEnd);
      };
    }
  }, [isDraggingPlayhead, zoom, onSeek]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if user is typing in an input field
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;

      // Delete/Backspace: Delete selected note
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        handleDeleteNote();
      }

      // Ctrl+C / Cmd+C: Copy selected note
      if (ctrlOrCmd && e.key === 'c') {
        e.preventDefault();
        handleCopyNote();
      }

      // Ctrl+V / Cmd+V: Paste note at playhead
      if (ctrlOrCmd && e.key === 'v') {
        e.preventDefault();
        handlePasteNote();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNote, copiedNote, currentTime]);

  const maxDuration = tracks.reduce((max, track) => {
    const notes = parseNotesFromDSL(track.id);
    const trackDuration = notes.reduce((sum, note) => Math.max(sum, note.start + note.duration), 0);
    return Math.max(max, trackDuration);
  }, 10);

  // Add padding to ensure all notes are visible and scrollable
  const totalWidth = (maxDuration + 4) * zoom; // Add 4 beats of padding

  return (
    <div className="w-full h-full bg-gray-950 border border-white/10 rounded-xl overflow-hidden flex flex-col relative">
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-gray-900 border border-white/20 rounded-lg p-6 flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-purple-600 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-white font-medium">Loading audio...</p>
            <p className="text-xs text-gray-400">Preparing samples and instruments</p>
          </div>
        </div>
      )}

      <div className="flex-none bg-gray-900 border-b border-white/10 p-4 flex items-center justify-between">
        <h3 className="text-white font-semibold">Timeline Editor</h3>
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-400">
            {selectedNote && "Press Delete to remove note"}
          </div>

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

      {/* Time ruler and track rows container */}
      <div className="flex flex-1 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-gray-900 [&::-webkit-scrollbar-thumb]:bg-gray-700 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:border [&::-webkit-scrollbar-thumb]:border-gray-800 hover:[&::-webkit-scrollbar-thumb]:bg-gray-600">
        {/* Fixed track labels column */}
        <div className="flex-shrink-0">
          <div className="w-48 h-8 bg-gray-900 border-r border-white/10"></div>
          {tracks.map((track) => (
            <div key={track.id} className="w-48 h-20 bg-gray-900 border-r border-b border-white/5 p-4">
              <div className="text-white font-medium">{track.id}</div>
              <div className="text-xs text-gray-500">{track.instrument?.split('/').pop()}</div>
            </div>
          ))}
        </div>

        {/* Scrollable timeline area */}
        <div
          className={`flex-1 overflow-x-auto overflow-y-hidden [&::-webkit-scrollbar]:h-2 [&::-webkit-scrollbar-track]:bg-gray-900 [&::-webkit-scrollbar-thumb]:bg-gray-700 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:border [&::-webkit-scrollbar-thumb]:border-gray-800 hover:[&::-webkit-scrollbar-thumb]:bg-gray-600 select-none ${isLoading ? 'cursor-not-allowed' : isDraggingPlayhead ? 'cursor-grabbing' : ''}`}
          ref={timelineRef}
          onMouseDown={handleTimelineClick}
        >
          <div className="relative" style={{ width: `${totalWidth}px`, minWidth: `${totalWidth}px` }}>
            {/* Global playback cursor - spans entire height */}
            <div
              id="red-line"
              className={`absolute top-0 bottom-0 w-[2px] bg-red-500 z-10 hover:bg-red-400 transition-colors ${isDraggingPlayhead ? 'cursor-grabbing bg-red-400' : 'cursor-grab'}`}
              style={{ left: `${secondsToBeats(currentTime) * zoom}px` }}
            />

            {/* Time ruler */}
            <div className="relative h-8 bg-gray-900 border-b border-white/10">
              {(() => {
                const { subdivision } = getGridSubdivision(zoom);
                const totalBeats = Math.ceil(maxDuration);
                const markers: JSX.Element[] = [];

                for (let beat = 0; beat <= totalBeats; beat += subdivision) {
                  // Round to avoid floating point errors
                  const roundedBeat = Math.round(beat / subdivision) * subdivision;
                  const bar = Math.floor(roundedBeat / 4) + 1;
                  const beatInBar = (roundedBeat % 4);
                  const isMeasureStart = Math.abs(roundedBeat % 4) < 0.001;
                  const isBeatStart = Math.abs(roundedBeat % 1) < 0.001;

                  let borderClass = '';
                  let labelContent = null;

                  // Highlight measure starts with thicker border
                  if (isMeasureStart) {
                    borderClass = 'border-l-2 border-gray-500';
                    labelContent = (
                      <span className="text-xs font-bold text-gray-200 absolute top-0.5 left-1">
                        {bar}
                      </span>
                    );
                  } else if (isBeatStart && subdivision < 1) {
                    // Show beat labels only when subdivision is smaller than a beat
                    borderClass = 'border-l border-gray-600';
                    labelContent = (
                      <span className="text-xs text-gray-400 absolute top-0.5 left-1">
                        {Math.floor(beatInBar) + 1}
                      </span>
                    );
                  } else {
                    // Regular subdivision line
                    borderClass = 'border-l border-gray-700/50';
                  }

                  markers.push(
                    <div
                      key={beat}
                      className={`absolute top-0 h-full ${borderClass}`}
                      style={{ left: `${beat * zoom}px` }}
                    >
                      {labelContent}
                    </div>
                  );
                }

                return markers;
              })()}
            </div>

            {/* Track timelines */}
            {tracks.map((track) => {
              const notes = parseNotesFromDSL(track.id);
              return (
                <div key={track.id} className="relative h-20 bg-gray-950 border-b border-white/5">
                  {/* Grid lines */}
                  {(() => {
                    const { subdivision } = getGridSubdivision(zoom);
                    const totalBeats = Math.ceil(maxDuration);
                    const gridLines: JSX.Element[] = [];

                    // Draw grid lines at snap subdivision interval
                    for (let beat = 0; beat <= totalBeats; beat += subdivision) {
                      // Round to avoid floating point errors
                      const roundedBeat = Math.round(beat / subdivision) * subdivision;
                      const isMeasureStart = Math.abs(roundedBeat % 4) < 0.001;
                      const isBeatStart = Math.abs(roundedBeat % 1) < 0.001;

                      let borderClass = '';

                      // Highlight measure starts with stronger border
                      if (isMeasureStart) {
                        borderClass = 'border-l border-gray-700';
                      } else if (isBeatStart && subdivision < 1) {
                        borderClass = 'border-l border-gray-800/70';
                      } else {
                        borderClass = 'border-l border-gray-800/40';
                      }

                      gridLines.push(
                        <div
                          key={beat}
                          className={`absolute top-0 h-full ${borderClass}`}
                          style={{ left: `${beat * zoom}px` }}
                        />
                      );
                    }

                    return gridLines;
                  })()}

                  {/* Notes */}
                  {notes.map((note, idx) => (
                    <div
                      key={idx}
                      className={`absolute top-2 bottom-2 ${note.isChord
                        ? 'bg-blue-600 hover:bg-blue-500'
                        : 'bg-purple-600 hover:bg-purple-500'
                        } rounded transition-colors ${selectedNote?.trackId === track.id && selectedNote?.noteIndex === idx
                          ? 'ring-2 ring-white'
                          : ''
                        } ${draggingNote || resizingNote ? 'cursor-grabbing' : 'cursor-grab'}`}
                      style={{
                        left: `${note.start * zoom}px`,
                        width: `${note.duration * zoom}px`,
                      }}
                      onMouseDown={(e) => handleNoteMouseDown(e, track.id, idx, false)}
                    >
                      <div className="text-xs text-white p-1 truncate pointer-events-none">
                        {pitchToNote(note.pitch)}{note.isChord ? ' ♫' : ''}
                      </div>

                      {/* Resize handle */}
                      <div
                        className="absolute right-0 top-0 bottom-0 w-2 bg-white/30 hover:bg-white/50 cursor-ew-resize"
                        onMouseDown={(e) => handleNoteMouseDown(e, track.id, idx, true)}
                      />
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}