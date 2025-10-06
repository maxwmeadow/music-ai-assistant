"use client";

import { useState, useRef, useEffect } from "react";
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
}

export function Timeline({ tracks, dslCode, onCodeChange, isPlaying, currentTime }: TimelineProps) {
  const [zoom, setZoom] = useState(50);
  const [selectedNote, setSelectedNote] = useState<{trackId: string, noteIndex: number} | null>(null);
  const [draggingNote, setDraggingNote] = useState<{trackId: string, noteIndex: number, startX: number, initialStart: number} | null>(null);
  const [resizingNote, setResizingNote] = useState<{trackId: string, noteIndex: number, startX: number, initialDuration: number} | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  const parseNotesFromDSL = (trackId: string): TimelineNote[] => {
    const trackMatch = dslCode.match(new RegExp(`track\\("${trackId}"\\)\\s*{([^}]+)}`, 's'));
    if (!trackMatch) return [];

    const trackContent = trackMatch[1];
    const notes: TimelineNote[] = [];
    let currentTime = 0;

    // Parse regular notes
    const noteMatches = trackContent.matchAll(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/g);
    for (const match of noteMatches) {
      const [, noteName, duration, velocity] = match;
      const pitch = noteToPitch(noteName);

      notes.push({
        pitch,
        start: currentTime,
        duration: parseFloat(duration),
        velocity: parseFloat(velocity)
      });

      currentTime += parseFloat(duration);
    }

    // Parse chords
    const chordMatches = trackContent.matchAll(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+)\)/g);
    for (const match of chordMatches) {
      const [, notesStr, duration, velocity] = match;
      const chordNotes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));
      const rootPitch = noteToPitch(chordNotes[0]);

      notes.push({
        pitch: rootPitch,
        start: currentTime,
        duration: parseFloat(duration),
        velocity: parseFloat(velocity),
        isChord: true
      });

      currentTime += parseFloat(duration);
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
    // Reconstruct DSL for this track
    const trackMatch = dslCode.match(new RegExp(`(track\\("${trackId}"\\)\\s*{)([^}]+)(})`, 's'));
    if (!trackMatch) return;

    const [fullMatch, opening, , closing] = trackMatch;

    // Keep instrument line
    const instrumentMatch = trackMatch[2].match(/instrument\("([^"]+)"\)/);
    const instrumentLine = instrumentMatch ? `  instrument("${instrumentMatch[1]}")\n` : '';

    // Generate new note lines
    const noteLines = updatedNotes.map(note => {
      const noteName = pitchToNote(note.pitch);
      const duration = note.duration.toFixed(1);
      const velocity = note.velocity.toFixed(1);

      if (note.isChord) {
        // For chords, we'd need to store original chord notes - for now just use root
        return `  chord(["${noteName}"], ${duration}, ${velocity})`;
      }
      return `  note("${noteName}", ${duration}, ${velocity})`;
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
      const deltaTime = deltaX / zoom;
      const newStart = Math.max(0, draggingNote.initialStart + deltaTime);

      const notes = parseNotesFromDSL(draggingNote.trackId);
      notes[draggingNote.noteIndex].start = newStart;

      // Recalculate durations to maintain spacing
      notes.sort((a, b) => a.start - b.start);
      for (let i = 0; i < notes.length - 1; i++) {
        const gap = notes[i + 1].start - notes[i].start;
        notes[i].duration = Math.max(0.1, gap);
      }

      updateDSLWithNewNotes(draggingNote.trackId, notes);
    } else if (resizingNote) {
      const deltaX = e.clientX - resizingNote.startX;
      const deltaDuration = deltaX / zoom;
      const newDuration = Math.max(0.1, resizingNote.initialDuration + deltaDuration);

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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        handleDeleteNote();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNote]);

  const maxDuration = tracks.reduce((max, track) => {
    const notes = parseNotesFromDSL(track.id);
    const trackDuration = notes.reduce((sum, note) => Math.max(sum, note.start + note.duration), 0);
    return Math.max(max, trackDuration);
  }, 10);

  return (
    <div className="bg-gray-950 border border-white/10 rounded-xl overflow-hidden">
      <div className="bg-gray-900 border-b border-white/10 p-4 flex items-center justify-between">
        <h3 className="text-white font-semibold">Timeline Editor</h3>
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-400">
            {selectedNote && "Press Delete to remove note"}
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

      <div className="bg-gray-900 border-b border-white/10 px-4 py-2 relative" style={{ paddingLeft: '200px' }}>
        <div className="relative h-6" style={{ width: `${maxDuration * zoom}px` }}>
          {Array.from({ length: Math.ceil(maxDuration) + 1 }).map((_, i) => (
            <div
              key={i}
              className="absolute top-0 h-full border-l border-gray-700"
              style={{ left: `${i * zoom}px` }}
            >
              <span className="text-xs text-gray-500 absolute top-0 -translate-x-1/2">
                {i}s
              </span>
            </div>
          ))}
          {isPlaying && (
            <div
              className="absolute top-0 w-0.5 h-full bg-red-500 z-10"
              style={{ left: `${currentTime * zoom}px` }}
            />
          )}
        </div>
      </div>

      <div className="overflow-auto max-h-96" ref={timelineRef}>
        {tracks.map((track) => {
          const notes = parseNotesFromDSL(track.id);
          return (
            <div key={track.id} className="flex border-b border-white/5">
              <div className="w-48 bg-gray-900 border-r border-white/10 p-4 flex-shrink-0">
                <div className="text-white font-medium">{track.id}</div>
                <div className="text-xs text-gray-500">{track.instrument?.split('/').pop()}</div>
              </div>

              <div className="relative flex-1 h-20 bg-gray-950" style={{ width: `${maxDuration * zoom}px` }}>
                {Array.from({ length: Math.ceil(maxDuration) + 1 }).map((_, i) => (
                  <div
                    key={i}
                    className="absolute top-0 h-full border-l border-gray-800/50"
                    style={{ left: `${i * zoom}px` }}
                  />
                ))}

                {notes.map((note, idx) => (
                  <div
                    key={idx}
                    className={`absolute top-2 bottom-2 ${
                      note.isChord 
                        ? 'bg-blue-600 hover:bg-blue-500' 
                        : 'bg-purple-600 hover:bg-purple-500'
                    } rounded transition-colors ${
                      selectedNote?.trackId === track.id && selectedNote?.noteIndex === idx
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

                {isPlaying && (
                  <div
                    className="absolute top-0 w-0.5 h-full bg-red-500 z-10 pointer-events-none"
                    style={{ left: `${currentTime * zoom}px` }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}