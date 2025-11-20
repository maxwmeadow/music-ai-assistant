"use client";

import { useState, useRef, useEffect, JSX } from "react";
import { ParsedTrack } from "@/lib/dslParser";
import {
  getTempoFromDSL,
  beatsToSeconds,
  secondsToBeats,
  getGridSubdivision,
  pitchToNote,
  parseNotesFromDSL
} from "./timelineHelpers";
import {
  handleDeleteNotes,
  handleCopyNotes,
  handlePasteNotes,
  handleNoteDrag,
  handleNoteResize,
  handleMultiNoteDrag,
  calculateNoteDrag,
  calculateNoteResize
} from "./timelineHandlers";
import {
  TimelineNote,
  SelectedNote,
  DraggingNote,
  ResizingNote,
  CopiedNotes,
  SNAP_OPTIONS
} from "./types";

interface TimelineProps {
  tracks: ParsedTrack[];
  dslCode: string;
  onCodeChange: (newCode: string) => void;
  isPlaying: boolean;
  currentTime: number;
  onSeek?: (time: number) => void;
  isLoading?: boolean;
  loopEnabled?: boolean;
  loopStart?: number;
  loopEnd?: number;
  onLoopChange?: (start: number, end: number) => void;
}

export function Timeline({
  tracks,
  dslCode,
  onCodeChange,
  isPlaying,
  currentTime,
  onSeek,
  isLoading = false,
  loopEnabled = false,
  loopStart = 0,
  loopEnd = 4,
  onLoopChange
}: TimelineProps) {
  const [zoom, setZoom] = useState(50);
  const [selectedNotes, setSelectedNotes] = useState<SelectedNote[]>([]);
  const [lastSelectedNote, setLastSelectedNote] = useState<SelectedNote | null>(null); // For shift-select range
  const [draggingNote, setDraggingNote] = useState<DraggingNote | null>(null);
  const [resizingNote, setResizingNote] = useState<ResizingNote | null>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const [isDraggingLoop, setIsDraggingLoop] = useState(false);
  const [loopDragStartX, setLoopDragStartX] = useState(0);
  const [loopDragStartTime, setLoopDragStartTime] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);
  const loopBarRef = useRef<HTMLDivElement>(null);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [snapValue, setSnapValue] = useState(0.25);
  const [copiedNotes, setCopiedNotes] = useState<CopiedNotes | null>(null);

  const tempo = getTempoFromDSL(dslCode);

  // Clear visual state only AFTER DSL has been updated with the correct position
  useEffect(() => {
    if (draggingNote?.committed && draggingNote.currentStart !== undefined) {
      // Verify the DSL has actually been updated with the new position
      const notes = parseNotesFromDSL(dslCode, draggingNote.trackId, tempo);
      const note = notes[draggingNote.noteIndex];

      // Only clear visual state if the DSL matches our expected position (within tolerance)
      if (note && Math.abs(note.start - draggingNote.currentStart) < 0.01) {
        requestAnimationFrame(() => {
          setDraggingNote(null);
        });
      }
    } else if (resizingNote?.committed && resizingNote.currentDuration !== undefined) {
      // Verify the DSL has actually been updated with the new duration
      const notes = parseNotesFromDSL(dslCode, resizingNote.trackId, tempo);
      const note = notes[resizingNote.noteIndex];

      // Only clear visual state if the DSL matches our expected duration (within tolerance)
      if (note && Math.abs(note.duration - resizingNote.currentDuration) < 0.01) {
        requestAnimationFrame(() => {
          setResizingNote(null);
        });
      }
    }
  }, [dslCode, draggingNote, resizingNote]);

  const handleNoteMouseDown = (e: React.MouseEvent, trackId: string, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const note = notes[noteIndex];

    // Check if this is a loop-generated note (read-only)
    if (note?.isFromLoop) {
      alert('This note is from a loop block and cannot be edited in the timeline.\n\nTo edit it, modify the loop in the code editor.');
      return;
    }

    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;
    const shift = e.shiftKey;
    const clickedNote = { trackId, noteIndex };
    const isAlreadySelected = selectedNotes.some(n => n.trackId === trackId && n.noteIndex === noteIndex);

    // Handle selection BEFORE setting drag state
    if (ctrlOrCmd) {
      // Ctrl/Cmd+Click: Toggle selection
      if (isAlreadySelected) {
        setSelectedNotes(selectedNotes.filter(n => !(n.trackId === trackId && n.noteIndex === noteIndex)));
      } else {
        setSelectedNotes([...selectedNotes, clickedNote]);
      }
      setLastSelectedNote(clickedNote);
    } else if (shift && lastSelectedNote && lastSelectedNote.trackId === trackId) {
      // Shift+Click: Select range (only within same track)
      const start = Math.min(lastSelectedNote.noteIndex, noteIndex);
      const end = Math.max(lastSelectedNote.noteIndex, noteIndex);
      const rangeSelection: SelectedNote[] = [];
      for (let i = start; i <= end; i++) {
        rangeSelection.push({ trackId, noteIndex: i });
      }
      setSelectedNotes(rangeSelection);
    } else if (!isAlreadySelected) {
      // Regular click on unselected note: Select single note
      setSelectedNotes([clickedNote]);
      setLastSelectedNote(clickedNote);
    }
    // If clicking an already-selected note without modifiers, keep selection (for multi-drag)

    if (isResize) {
      setResizingNote({
        trackId,
        noteIndex,
        startX: e.clientX,
        initialDuration: notes[noteIndex].duration
      });
    } else {
      // Store all selected notes for multi-drag
      setDraggingNote({
        trackId,
        noteIndex,
        startX: e.clientX,
        initialStart: notes[noteIndex].start,
        selectedNotes: isAlreadySelected && selectedNotes.length > 1 ? selectedNotes : undefined
      });
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (draggingNote && !draggingNote.committed) {
      // Only update if not committed (still actively dragging)
      const deltaX = e.clientX - draggingNote.startX;
      const newStart = calculateNoteDrag(deltaX, zoom, draggingNote.initialStart, snapValue, snapEnabled);

      setDraggingNote({
        ...draggingNote,
        currentStart: newStart
      });
    } else if (resizingNote && !resizingNote.committed) {
      // Only update if not committed (still actively resizing)
      const deltaX = e.clientX - resizingNote.startX;
      const newDuration = calculateNoteResize(deltaX, zoom, resizingNote.initialDuration, snapValue, snapEnabled);

      setResizingNote({
        ...resizingNote,
        currentDuration: newDuration
      });
    }
  };

  const handleMouseUp = () => {
    // Mark as committed (stops mouse updates but keeps visual state until DSL updates)
    if (draggingNote && draggingNote.currentStart !== undefined) {
      const deltaBeats = draggingNote.currentStart - draggingNote.initialStart;

      // Mark as committed to stop further mouse updates
      setDraggingNote({
        ...draggingNote,
        committed: true
      });

      // Apply DSL changes (visual state will be cleared by useEffect when dslCode updates)
      if (draggingNote.selectedNotes && draggingNote.selectedNotes.length > 1) {
        handleMultiNoteDrag(deltaBeats, draggingNote.selectedNotes, dslCode, tempo, snapValue, snapEnabled, onCodeChange);
      } else {
        handleNoteDrag(deltaBeats * zoom, zoom, draggingNote, dslCode, tempo, snapValue, snapEnabled, onCodeChange);
      }
    } else if (resizingNote && resizingNote.currentDuration !== undefined) {
      const deltaBeats = resizingNote.currentDuration - resizingNote.initialDuration;

      // Mark as committed to stop further mouse updates
      setResizingNote({
        ...resizingNote,
        committed: true
      });

      // Apply DSL changes (visual state will be cleared by useEffect when dslCode updates)
      handleNoteResize(deltaBeats * zoom, zoom, resizingNote, dslCode, tempo, snapValue, snapEnabled, onCodeChange);
    }
  };

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (isLoading) return;
    if (draggingNote || resizingNote) return;
    if (!onSeek || !timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left + timelineRef.current.scrollLeft;
    const clickedBeats = clickX / zoom;
    const clickedTime = beatsToSeconds(clickedBeats, tempo);

    onSeek(Math.max(0, clickedTime));
    setIsDraggingPlayhead(true);
    setSelectedNotes([]);
  };

  const handlePlayheadDrag = (e: MouseEvent) => {
    if (!isDraggingPlayhead || !onSeek || !timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const dragX = e.clientX - rect.left + timelineRef.current.scrollLeft;
    const draggedBeats = dragX / zoom;
    const draggedTime = beatsToSeconds(draggedBeats, tempo);

    onSeek(Math.max(0, draggedTime));
  };

  const handlePlayheadDragEnd = () => {
    setIsDraggingPlayhead(false);
  };

  // Loop region drag handlers
  const handleLoopBarMouseDown = (e: React.MouseEvent) => {
    if (!onLoopChange || !loopBarRef.current) return;
    e.stopPropagation();

    const rect = loopBarRef.current.getBoundingClientRect();
    // Loop bar is inside scrolling content, so rect.left already accounts for scroll
    const clickX = e.clientX - rect.left;
    const clickedBeats = clickX / zoom;
    const clickedTime = beatsToSeconds(clickedBeats, tempo);

    setLoopDragStartX(clickX);
    setLoopDragStartTime(clickedTime);
    setIsDraggingLoop(true);
    onLoopChange(clickedTime, clickedTime);
  };

  const handleLoopDrag = (e: MouseEvent) => {
    if (!isDraggingLoop || !onLoopChange || !loopBarRef.current || !timelineRef.current) return;

    const rect = loopBarRef.current.getBoundingClientRect();
    // Loop bar is inside scrolling content, so rect.left already accounts for scroll
    const currentX = e.clientX - rect.left;
    const currentBeats = currentX / zoom;
    const currentTime = beatsToSeconds(currentBeats, tempo);

    const start = Math.min(loopDragStartTime, currentTime);
    const end = Math.max(loopDragStartTime, currentTime);

    onLoopChange(Math.max(0, start), Math.max(0, end));
  };

  const handleLoopDragEnd = () => {
    setIsDraggingLoop(false);
  };

  // Mouse event listeners
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

  useEffect(() => {
    if (isDraggingLoop) {
      window.addEventListener('mousemove', handleLoopDrag);
      window.addEventListener('mouseup', handleLoopDragEnd);
      return () => {
        window.removeEventListener('mousemove', handleLoopDrag);
        window.removeEventListener('mouseup', handleLoopDragEnd);
      };
    }
  }, [isDraggingLoop, zoom, onLoopChange]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        handleDeleteNotes(selectedNotes, dslCode, tempo, onCodeChange, setSelectedNotes);
      }

      if (ctrlOrCmd && e.key === 'c') {
        e.preventDefault();
        handleCopyNotes(selectedNotes, dslCode, tempo, setCopiedNotes);
      }

      if (ctrlOrCmd && e.key === 'v') {
        e.preventDefault();
        handlePasteNotes(copiedNotes, currentTime, dslCode, tempo, snapValue, snapEnabled, onCodeChange, setSelectedNotes);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNotes, copiedNotes, currentTime]);

  // Auto-scroll to keep playhead visible (like DAWs)
  useEffect(() => {
    if (!timelineRef.current || !isPlaying) return;

    const container = timelineRef.current;
    const playheadPosition = secondsToBeats(currentTime, tempo) * zoom;
    const viewportLeft = container.scrollLeft;
    const viewportRight = viewportLeft + container.clientWidth;
    const scrollMargin = 100; // Keep playhead this many pixels from edge

    // Check if playhead is going off the right edge
    if (playheadPosition > viewportRight - scrollMargin) {
      // Scroll to keep playhead at first 1/4 of viewport (more content ahead visible)
      container.scrollLeft = playheadPosition - container.clientWidth / 4;
    }
    // Check if playhead is going off the left edge (when looping back)
    else if (playheadPosition < viewportLeft + scrollMargin) {
      container.scrollLeft = Math.max(0, playheadPosition - container.clientWidth / 4);
    }
  }, [currentTime, isPlaying, zoom, tempo]);

  const maxDuration = tracks.reduce((max, track) => {
    const notes = parseNotesFromDSL(dslCode, track.id, tempo);
    const trackDuration = notes.reduce((sum, note) => Math.max(sum, note.start + note.duration), 0);
    return Math.max(max, trackDuration);
  }, 10);

  const totalWidth = (maxDuration + 4) * zoom;

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
            {selectedNotes.length > 0 && `Press Delete to remove ${selectedNotes.length} note${selectedNotes.length > 1 ? 's' : ''}`}
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
            <div key={track.id} className="w-48 h-20 bg-gray-900 border-r border-b border-white/5 p-2">
              <div className="text-white font-medium text-sm">{track.id}</div>
              <div className="text-xs text-gray-500">
                {track.instrument?.split('/').pop()}
              </div>
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
            {/* Global playback cursor */}
            <div
              id="red-line"
              className={`absolute top-0 bottom-0 w-[2px] bg-red-500 z-10 hover:bg-red-400 transition-colors ${isDraggingPlayhead ? 'cursor-grabbing bg-red-400' : 'cursor-grab'}`}
              style={{ left: `${secondsToBeats(currentTime, tempo) * zoom}px` }}
            />

            {/* Loop Region Bar */}
            {loopEnabled && (
              <div
                ref={loopBarRef}
                className="relative h-6 bg-gray-800 border-b border-white/5 cursor-crosshair"
                onMouseDown={handleLoopBarMouseDown}
              >
                {/* Loop region highlight */}
                <div
                  className="absolute top-0 bottom-0 bg-purple-600/30 border-l-2 border-r-2 border-purple-500"
                  style={{
                    left: `${secondsToBeats(loopStart, tempo) * zoom}px`,
                    width: `${secondsToBeats(loopEnd - loopStart, tempo) * zoom}px`,
                  }}
                >
                  <div className="absolute inset-0 flex items-center justify-center text-[10px] text-white font-semibold">
                    LOOP
                  </div>
                </div>
              </div>
            )}

            {/* Time ruler */}
            <div className="relative h-8 bg-gray-900 border-b border-white/10">
              {(() => {
                const { subdivision } = getGridSubdivision(zoom, snapValue, snapEnabled);
                const totalBeats = Math.ceil(maxDuration);
                const markers: JSX.Element[] = [];

                for (let beat = 0; beat <= totalBeats; beat += subdivision) {
                  const roundedBeat = Math.round(beat / subdivision) * subdivision;
                  const bar = Math.floor(roundedBeat / 4) + 1;
                  const beatInBar = (roundedBeat % 4);
                  const isMeasureStart = Math.abs(roundedBeat % 4) < 0.001;
                  const isBeatStart = Math.abs(roundedBeat % 1) < 0.001;

                  let borderClass = '';
                  let labelContent = null;

                  if (isMeasureStart) {
                    borderClass = 'border-l-2 border-gray-500';
                    labelContent = (
                      <span className="text-xs font-bold text-gray-200 absolute top-0.5 left-1">
                        {bar}
                      </span>
                    );
                  } else if (isBeatStart && subdivision < 1) {
                    borderClass = 'border-l border-gray-600';
                    labelContent = (
                      <span className="text-xs text-gray-400 absolute top-0.5 left-1">
                        {Math.floor(beatInBar) + 1}
                      </span>
                    );
                  } else {
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
              const notes = parseNotesFromDSL(dslCode, track.id, tempo);

              return (
                <div key={track.id} className="relative h-20 bg-gray-950 border-b border-white/5">
                  {/* Grid lines */}
                  {(() => {
                    const { subdivision } = getGridSubdivision(zoom, snapValue, snapEnabled);
                    const totalBeats = Math.ceil(maxDuration);
                    const gridLines: JSX.Element[] = [];

                    for (let beat = 0; beat <= totalBeats; beat += subdivision) {
                      const roundedBeat = Math.round(beat / subdivision) * subdivision;
                      const isMeasureStart = Math.abs(roundedBeat % 4) < 0.001;
                      const isBeatStart = Math.abs(roundedBeat % 1) < 0.001;

                      let borderClass = '';

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
                      className={`absolute top-2 bottom-2 ${
                        note.isFromLoop
                          ? 'bg-gray-600 opacity-50' // Loop notes: gray and semi-transparent
                          : note.isChord
                            ? 'bg-blue-600 hover:bg-blue-500'
                            : 'bg-purple-600 hover:bg-purple-500'
                        } rounded transition-colors ${selectedNotes.some(n => n.trackId === track.id && n.noteIndex === idx)
                          ? 'ring-2 ring-white'
                          : ''
                        } ${
                          note.isFromLoop
                            ? 'cursor-not-allowed' // Loop notes: not editable
                            : draggingNote || resizingNote
                              ? 'cursor-grabbing'
                              : 'cursor-grab'
                        }`}
                      style={{
                        left: `${(() => {
                          // Check if this note is being dragged (either primary or part of multi-selection)
                          const isPrimaryDrag = draggingNote?.trackId === track.id && draggingNote?.noteIndex === idx;
                          const isMultiDrag = draggingNote?.selectedNotes?.some(n => n.trackId === track.id && n.noteIndex === idx);

                          if ((isPrimaryDrag || isMultiDrag) && draggingNote && draggingNote.currentStart !== undefined) {
                            const delta = draggingNote.currentStart - draggingNote.initialStart;
                            return (note.start + delta) * zoom;
                          }
                          return note.start * zoom;
                        })()}px`,
                        width: `${
                          resizingNote?.trackId === track.id && resizingNote?.noteIndex === idx && resizingNote.currentDuration !== undefined
                            ? resizingNote.currentDuration * zoom
                            : note.duration * zoom
                        }px`,
                      }}
                      onMouseDown={(e) => handleNoteMouseDown(e, track.id, idx, false)}
                    >
                      <div className="text-xs text-white p-1 truncate pointer-events-none">
                        {pitchToNote(note.pitch)}{note.isChord ? ' ♫' : ''}
                      </div>

                      {/* Resize handle - hidden for loop notes */}
                      {!note.isFromLoop && (
                        <div
                          className="absolute right-0 top-0 bottom-0 w-2 bg-white/30 hover:bg-white/50 cursor-ew-resize"
                          onMouseDown={(e) => handleNoteMouseDown(e, track.id, idx, true)}
                        />
                      )}
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
