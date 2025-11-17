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
  handleNoteResize
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
}

export function Timeline({ tracks, dslCode, onCodeChange, isPlaying, currentTime, onSeek, isLoading = false }: TimelineProps) {
  const [zoom, setZoom] = useState(50);
  const [selectedNotes, setSelectedNotes] = useState<SelectedNote[]>([]);
  const [lastSelectedNote, setLastSelectedNote] = useState<SelectedNote | null>(null); // For shift-select range
  const [draggingNote, setDraggingNote] = useState<DraggingNote | null>(null);
  const [resizingNote, setResizingNote] = useState<ResizingNote | null>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [snapValue, setSnapValue] = useState(0.25);
  const [copiedNotes, setCopiedNotes] = useState<CopiedNotes | null>(null);

  const tempo = getTempoFromDSL(dslCode);

  const handleNoteMouseDown = (e: React.MouseEvent, trackId: string, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;
    const shift = e.shiftKey;

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

    // Handle multi-selection
    const clickedNote = { trackId, noteIndex };

    if (ctrlOrCmd) {
      // Ctrl/Cmd+Click: Toggle selection
      const isSelected = selectedNotes.some(n => n.trackId === trackId && n.noteIndex === noteIndex);
      if (isSelected) {
        setSelectedNotes(selectedNotes.filter(n => !(n.trackId === trackId && n.noteIndex === noteIndex)));
      } else {
        setSelectedNotes([...selectedNotes, clickedNote]);
      }
      setLastSelectedNote(clickedNote);
    } else if (shift && lastSelectedNote && lastSelectedNote.trackId === trackId) {
      // Shift+Click: Select range (only within same track)
      const trackNotes = notes;
      const start = Math.min(lastSelectedNote.noteIndex, noteIndex);
      const end = Math.max(lastSelectedNote.noteIndex, noteIndex);
      const rangeSelection: SelectedNote[] = [];
      for (let i = start; i <= end; i++) {
        rangeSelection.push({ trackId, noteIndex: i });
      }
      setSelectedNotes(rangeSelection);
    } else {
      // Regular click: Select single note
      setSelectedNotes([clickedNote]);
      setLastSelectedNote(clickedNote);
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (draggingNote) {
      const deltaX = e.clientX - draggingNote.startX;
      handleNoteDrag(deltaX, zoom, draggingNote, dslCode, tempo, snapValue, snapEnabled, onCodeChange);
    } else if (resizingNote) {
      const deltaX = e.clientX - resizingNote.startX;
      handleNoteResize(deltaX, zoom, resizingNote, dslCode, tempo, snapValue, snapEnabled, onCodeChange);
    }
  };

  const handleMouseUp = () => {
    setDraggingNote(null);
    setResizingNote(null);
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
            {/* Global playback cursor */}
            <div
              id="red-line"
              className={`absolute top-0 bottom-0 w-[2px] bg-red-500 z-10 hover:bg-red-400 transition-colors ${isDraggingPlayhead ? 'cursor-grabbing bg-red-400' : 'cursor-grab'}`}
              style={{ left: `${secondsToBeats(currentTime, tempo) * zoom}px` }}
            />

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
                      className={`absolute top-2 bottom-2 ${note.isChord
                        ? 'bg-blue-600 hover:bg-blue-500'
                        : 'bg-purple-600 hover:bg-purple-500'
                        } rounded transition-colors ${selectedNotes.some(n => n.trackId === track.id && n.noteIndex === idx)
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
