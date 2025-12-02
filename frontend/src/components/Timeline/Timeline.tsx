"use client";

import { useState, useRef, useEffect, useMemo, JSX } from "react";
import { ParsedTrack } from "@/lib/dslParser";
import {
  getTempoFromDSL,
  beatsToSeconds,
  secondsToBeats,
  getGridSubdivision,
  pitchToNote,
  pitchToDrumName,
  isDrumTrack,
  parseNotesFromDSL,
  updateDSLWithNewNotes
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
import { Mic, Drum } from "lucide-react";

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
  onPlaybackStart?: () => void;
  onPlaybackStop?: () => void;
  onMelodyGenerated?: (ir: any) => void;
  onCompile?: () => Promise<void>;
  executableCode?: string;
  audioClips?: Array<{ id: string; trackId: string; url: string; start: number; duration: number }>;
  onUpdateAudioClip?: (clip: { id: string; start?: number; duration?: number }) => void;
  onDeleteAudioClip?: (id: string) => void;
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
  onLoopChange,
  onPlaybackStart,
  onPlaybackStop,
  onMelodyGenerated,
  onCompile,
  executableCode
  , audioClips,
  onUpdateAudioClip,
  onDeleteAudioClip
}: TimelineProps) {
  const [zoom, setZoom] = useState(50);
  const [selectedNotes, setSelectedNotes] = useState<SelectedNote[]>([]);
  const [lastSelectedNote, setLastSelectedNote] = useState<SelectedNote | null>(null); // For shift-select range
  const [draggingNote, setDraggingNote] = useState<DraggingNote | null>(null);
  const [resizingNote, setResizingNote] = useState<ResizingNote | null>(null);
  const [draggingAudio, setDraggingAudio] = useState<{
    id: string;
    trackId: string;
    startX: number;
    initialStart: number;
    currentStart?: number;
    committed?: boolean;
  } | null>(null);

  const [resizingAudio, setResizingAudio] = useState<{
    id: string;
    trackId: string;
    startX: number;
    initialDuration: number;
    currentDuration?: number;
    committed?: boolean;
  } | null>(null);

  const [selectedAudioId, setSelectedAudioId] = useState<string | null>(null);
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

  // Memoize parsed notes to prevent expensive re-parsing on every render
  const parsedNotesMap = useMemo(() => {
    const map = new Map<string, TimelineNote[]>();
    tracks.forEach(track => {
      map.set(track.id, parseNotesFromDSL(dslCode, track.id, tempo));
    });
    return map;
  }, [dslCode, tracks, tempo]);

  // Clear visual state only AFTER DSL has been updated with the correct position
  useEffect(() => {
    if (draggingNote?.committed && draggingNote.currentStart !== undefined) {
      // Verify the DSL has actually been updated with the new position
      const notes = parseNotesFromDSL(dslCode, draggingNote.trackId, tempo);
      const expectedStart = draggingNote.currentStart; // Extract for type narrowing

      // Check if ANY note in the track matches our expected position
      // (Note index may have changed due to re-ordering after DSL update)
      const matchingNote = notes.find(n => Math.abs(n.start - expectedStart) < 0.01);

      if (matchingNote) {
        requestAnimationFrame(() => {
          setDraggingNote(null);
        });
      } else {
        // Fallback: clear after 500ms even if DSL doesn't match
        // This prevents stuck drag states from note re-ordering edge cases
        const timeout = setTimeout(() => {
          setDraggingNote(null);
        }, 500);
        return () => clearTimeout(timeout);
      }
    } else if (resizingNote?.committed && resizingNote.currentDuration !== undefined) {
      // Verify the DSL has actually been updated with the new duration
      const notes = parseNotesFromDSL(dslCode, resizingNote.trackId, tempo);
      const expectedDuration = resizingNote.currentDuration; // Extract for type narrowing

      // Check if ANY note in the track matches our expected duration
      // (Note index may have changed due to re-ordering after DSL update)
      const matchingNote = notes.find(n => Math.abs(n.duration - expectedDuration) < 0.01);

      if (matchingNote) {
        requestAnimationFrame(() => {
          setResizingNote(null);
        });
      } else {
        // Fallback: clear after 500ms even if DSL doesn't match
        const timeout = setTimeout(() => {
          setResizingNote(null);
        }, 500);
        return () => clearTimeout(timeout);
      }
    }
  }, [dslCode, draggingNote, resizingNote]);

  const handleNoteMouseDown = (e: React.MouseEvent, trackId: string, noteIndex: number, isResize: boolean) => {
    e.stopPropagation();
    e.preventDefault(); // Prevent text selection
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const note = notes[noteIndex];

    // Check if this is a loop-generated note (read-only)
    if (note?.isFromLoop) {
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

  const handleAudioMouseDown = (e: React.MouseEvent, clipId: string, trackId: string, isResize: boolean) => {
    e.stopPropagation();
    e.preventDefault();
    const clip = (audioClips || []).find(c => c.id === clipId);
    if (!clip) return;

    const isAlreadySelected = selectedAudioId === clipId;
    if (!isAlreadySelected) {
      setSelectedNotes([]);
      setSelectedAudioId(clipId);
    }

    if (isResize) {
      setResizingAudio({ id: clipId, trackId, startX: e.clientX, initialDuration: clip.duration });
    } else {
      setDraggingAudio({ id: clipId, trackId, startX: e.clientX, initialStart: clip.start });
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (draggingNote && !draggingNote.committed) {
      e.preventDefault(); // Prevent text selection during drag

      // Only update if not committed (still actively dragging)
      const deltaX = e.clientX - draggingNote.startX;
      const newStart = calculateNoteDrag(deltaX, zoom, draggingNote.initialStart, snapValue, snapEnabled);

      setDraggingNote({
        ...draggingNote,
        currentStart: newStart
      });
    } else if (resizingNote && !resizingNote.committed) {
      e.preventDefault(); // Prevent text selection during drag

      // Only update if not committed (still actively resizing)
      const deltaX = e.clientX - resizingNote.startX;
      const newDuration = calculateNoteResize(deltaX, zoom, resizingNote.initialDuration, snapValue, snapEnabled);

      setResizingNote({
        ...resizingNote,
        currentDuration: newDuration
      });
    } else if (draggingAudio && !draggingAudio.committed) {
      e.preventDefault();
      const deltaX = e.clientX - draggingAudio.startX;
      const newStart = calculateNoteDrag(deltaX, zoom, draggingAudio.initialStart, snapValue, snapEnabled);
      setDraggingAudio({ ...draggingAudio, currentStart: newStart });
    } else if (resizingAudio && !resizingAudio.committed) {
      e.preventDefault();
      const deltaX = e.clientX - resizingAudio.startX;
      const newDuration = calculateNoteResize(deltaX, zoom, resizingAudio.initialDuration, snapValue, snapEnabled);
      setResizingAudio({ ...resizingAudio, currentDuration: newDuration });
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
    } else if (draggingNote) {
      // No drag happened, just clear the state immediately
      setDraggingNote(null);
    }

    if (resizingNote && resizingNote.currentDuration !== undefined) {
      // Mark as committed to stop further mouse updates
      setResizingNote({
        ...resizingNote,
        committed: true
      });

      // Apply the already-calculated duration directly (no need to recalculate)
      const notes = parseNotesFromDSL(dslCode, resizingNote.trackId, tempo);
      notes[resizingNote.noteIndex].duration = resizingNote.currentDuration;
      const newCode = updateDSLWithNewNotes(dslCode, resizingNote.trackId, notes, tempo);
      onCodeChange(newCode);
    } else if (resizingNote) {
      // No resize happened, just clear the state immediately
      setResizingNote(null);
    }

    // Audio drag commit
    if (draggingAudio && draggingAudio.currentStart !== undefined) {
      setDraggingAudio({ ...draggingAudio, committed: true });
      if (onUpdateAudioClip) onUpdateAudioClip({ id: draggingAudio.id, start: draggingAudio.currentStart });
    } else {
      setDraggingAudio(null);
    }

    if (resizingAudio && resizingAudio.currentDuration !== undefined) {
      setResizingAudio({ ...resizingAudio, committed: true });
      if (onUpdateAudioClip) onUpdateAudioClip({ id: resizingAudio.id, duration: resizingAudio.currentDuration });
    } else {
      setResizingAudio(null);
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
    setSelectedAudioId(null);
  };

  const handlePlayheadDrag = (e: MouseEvent) => {
    if (!isDraggingPlayhead || !onSeek || !timelineRef.current) return;
    e.preventDefault(); // Prevent text selection during drag

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
    e.preventDefault(); // Prevent text selection during drag

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
    if (draggingNote || resizingNote || draggingAudio || resizingAudio) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [draggingNote, resizingNote, draggingAudio, resizingAudio]);

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
        if (selectedNotes.length > 0) {
          handleDeleteNotes(selectedNotes, dslCode, tempo, onCodeChange, setSelectedNotes);
        } else if (selectedAudioId && onDeleteAudioClip) {
          onDeleteAudioClip(selectedAudioId);
          setSelectedAudioId(null);
        }
      }

      if (ctrlOrCmd && e.key === 'c') {
        e.preventDefault();
        handleCopyNotes(selectedNotes, dslCode, tempo, setCopiedNotes);
      }

      if (ctrlOrCmd && e.key === 'x') {
        e.preventDefault();
        // Cut: copy then delete
        handleCopyNotes(selectedNotes, dslCode, tempo, setCopiedNotes);
        handleDeleteNotes(selectedNotes, dslCode, tempo, onCodeChange, setSelectedNotes);
      }

      if (ctrlOrCmd && e.key === 'v') {
        e.preventDefault();
        handlePasteNotes(copiedNotes, currentTime, dslCode, tempo, snapValue, snapEnabled, onCodeChange, setSelectedNotes);
      }

      if (ctrlOrCmd && e.key === 'a') {
        e.preventDefault();
        // Select all editable notes (not loop-generated) across all tracks
        const allSelectableNotes: SelectedNote[] = [];
        tracks.forEach(track => {
          const notes = parseNotesFromDSL(dslCode, track.id, tempo);
          notes.forEach((note, idx) => {
            if (!note.isFromLoop) {
              allSelectableNotes.push({ trackId: track.id, noteIndex: idx });
            }
          });
        });
        setSelectedNotes(allSelectableNotes);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNotes, copiedNotes, currentTime, tracks, dslCode, tempo, snapValue, snapEnabled, onCodeChange, selectedAudioId, onDeleteAudioClip]);

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

  const maxDuration = useMemo(() => {
    return tracks.reduce((max, track) => {
      const notes = parsedNotesMap.get(track.id) || [];
      const trackDuration = notes.reduce((sum, note) => Math.max(sum, note.start + note.duration), 0);
      return Math.max(max, trackDuration);
    }, 10);
  }, [tracks, parsedNotesMap]);
  // Include audio clips in max duration calculation
  const adjustedMaxDuration = useMemo(() => {
    const clipsMax = (audioClips || []).reduce((m, c) => Math.max(m, c.start + c.duration), 0);
    return Math.max(maxDuration, clipsMax);
  }, [maxDuration, audioClips]);

  const totalWidth = (adjustedMaxDuration + 4) * zoom;

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
        <div className="flex items-center gap-3">
          <h3 className="text-white font-semibold">Timeline Editor</h3>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-400">
            {selectedNotes.length > 0 ? `Press Delete to remove ${selectedNotes.length} note${selectedNotes.length > 1 ? 's' : ''}` : (selectedAudioId ? 'Press Delete to remove selected audio clip' : '')}
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
          {(audioClips || []).map(c => (
            <div key={c.id} className="w-48 h-20 bg-gray-900 border-r border-b border-white/5 p-2">
              <div className="text-white font-medium text-sm">{c.id}</div>
              <div className="text-xs text-gray-500">Vocal</div>
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

                // CSS grid background
                const gridBackground = (
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      backgroundImage: `
                        repeating-linear-gradient(
                          90deg,
                          rgb(107, 114, 128) 0px,
                          rgb(107, 114, 128) 2px,
                          transparent 2px,
                          transparent ${4 * zoom}px
                        ),
                        repeating-linear-gradient(
                          90deg,
                          rgb(75, 85, 99) 0px,
                          rgb(75, 85, 99) 1px,
                          transparent 1px,
                          transparent ${subdivision < 1 ? zoom : 4 * zoom}px
                        ),
                        repeating-linear-gradient(
                          90deg,
                          rgba(55, 65, 81, 0.5) 0px,
                          rgba(55, 65, 81, 0.5) 1px,
                          transparent 1px,
                          transparent ${subdivision * zoom}px
                        )
                      `,
                      backgroundSize: `${totalWidth}px 100%`,
                      backgroundPosition: '0 0'
                    }}
                  />
                );

                // Only create JSX for measure/beat labels (not grid lines!)
                const labels: JSX.Element[] = [];
                for (let beat = 0; beat <= totalBeats; beat += subdivision) {
                  const roundedBeat = Math.round(beat / subdivision) * subdivision;
                  const bar = Math.floor(roundedBeat / 4) + 1;
                  const beatInBar = (roundedBeat % 4);
                  const isMeasureStart = Math.abs(roundedBeat % 4) < 0.001;
                  const isBeatStart = Math.abs(roundedBeat % 1) < 0.001;

                  if (isMeasureStart) {
                    labels.push(
                      <span
                        key={beat}
                        className="text-xs font-bold text-gray-200 absolute top-0.5 pointer-events-none"
                        style={{ left: `${beat * zoom + 4}px` }}
                      >
                        {bar}
                      </span>
                    );
                  } else if (isBeatStart && subdivision < 1) {
                    labels.push(
                      <span
                        key={beat}
                        className="text-xs text-gray-400 absolute top-0.5 pointer-events-none"
                        style={{ left: `${beat * zoom + 4}px` }}
                      >
                        {Math.floor(beatInBar) + 1}
                      </span>
                    );
                  }
                }

                return (
                  <>
                    {gridBackground}
                    {labels}
                  </>
                );
              })()}
            </div>

            {/* Track timelines */}
            {tracks.map((track) => {
              const notes = parsedNotesMap.get(track.id) || [];
              const { subdivision } = getGridSubdivision(zoom, snapValue, snapEnabled);

              return (
                <div key={track.id} className="relative h-20 bg-gray-950 border-b border-white/5">
                  {/* Grid lines - CSS background pattern (no DOM elements!) */}
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      backgroundImage: `
                        repeating-linear-gradient(
                          90deg,
                          rgb(55, 65, 81) 0px,
                          rgb(55, 65, 81) 1px,
                          transparent 1px,
                          transparent ${4 * zoom}px
                        ),
                        repeating-linear-gradient(
                          90deg,
                          rgba(55, 65, 81, 0.7) 0px,
                          rgba(55, 65, 81, 0.7) 1px,
                          transparent 1px,
                          transparent ${subdivision < 1 ? zoom : 4 * zoom}px
                        ),
                        repeating-linear-gradient(
                          90deg,
                          rgba(55, 65, 81, 0.4) 0px,
                          rgba(55, 65, 81, 0.4) 1px,
                          transparent 1px,
                          transparent ${subdivision * zoom}px
                        )
                      `,
                      backgroundSize: `${totalWidth}px 100%`,
                      backgroundPosition: '0 0'
                    }}
                  />

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

                          if (draggingNote && draggingNote.currentStart !== undefined) {
                            if (isPrimaryDrag) {
                              // Primary drag: use the exact dragged position
                              return draggingNote.currentStart * zoom;
                            } else if (isMultiDrag) {
                              // Multi-drag: calculate delta and apply to this note
                              const delta = draggingNote.currentStart - draggingNote.initialStart;
                              return (note.start + delta) * zoom;
                            }
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
                        {(() => {
                          const isDrum = isDrumTrack(track.id, track.instrument || undefined);
                          const noteWidth = (resizingNote?.trackId === track.id && resizingNote?.noteIndex === idx && resizingNote.currentDuration !== undefined
                            ? resizingNote.currentDuration * zoom
                            : note.duration * zoom);

                          // Smart collapsing: show all chord notes only if wide enough
                          const minWidthForFullChord = 80; // Minimum width in px to show all chord notes

                          if (note.isChord && note.chordPitches && noteWidth >= minWidthForFullChord) {
                            // Wide enough: show all chord notes
                            const allNotes = note.chordPitches
                              .map(p => isDrum ? pitchToDrumName(p) : pitchToNote(p))
                              .join(', ');
                            return `${allNotes}${note.isFromLoop ? ' 🔒' : ''}`;
                          } else {
                            // Too narrow or single note: show root only
                            const noteName = isDrum ? pitchToDrumName(note.pitch) : pitchToNote(note.pitch);
                            return `${noteName}${note.isChord ? ' ♫' : ''}${note.isFromLoop ? ' 🔒' : ''}`;
                          }
                        })()}
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

                  {/* (audio clips are rendered in dedicated vocal rows below) */}
                </div>
              );
            })}

            {/* Vocal/audio clip rows (each clip gets its own track row) */}
            {(audioClips || []).map((clip) => {
              const subdivisionLocal = getGridSubdivision(zoom, snapValue, snapEnabled).subdivision;
              return (
                <div key={clip.id} className="relative h-20 bg-gray-950 border-b border-white/5">
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      backgroundImage: `
                        repeating-linear-gradient(
                          90deg,
                          rgb(55, 65, 81) 0px,
                          rgb(55, 65, 81) 1px,
                          transparent 1px,
                          transparent ${4 * zoom}px
                        ),
                        repeating-linear-gradient(
                          90deg,
                          rgba(55, 65, 81, 0.7) 0px,
                          rgba(55, 65, 81, 0.7) 1px,
                          transparent 1px,
                          transparent ${subdivisionLocal < 1 ? zoom : 4 * zoom}px
                        ),
                        repeating-linear-gradient(
                          90deg,
                          rgba(55, 65, 81, 0.4) 0px,
                          rgba(55, 65, 81, 0.4) 1px,
                          transparent 1px,
                          transparent ${subdivisionLocal * zoom}px
                        )
                      `,
                      backgroundSize: `${totalWidth}px 100%`,
                      backgroundPosition: '0 0'
                    }}
                  />

                  {/* Render the clip itself */}
                  <div
                    key={`${clip.id}-clip`}
                    className={`absolute top-4 bottom-4 bg-green-600 hover:bg-green-500 rounded transition-colors ${selectedAudioId === clip.id ? 'ring-2 ring-white' : ''} ${draggingNote || resizingNote ? 'cursor-grabbing' : 'cursor-grab'}`}
                    style={{
                      left: `${((draggingAudio && draggingAudio.id === clip.id && draggingAudio.currentStart !== undefined) ? draggingAudio.currentStart : clip.start) * zoom}px`,
                      width: `${(resizingAudio && resizingAudio.id === clip.id && resizingAudio.currentDuration !== undefined) ? resizingAudio.currentDuration * zoom : clip.duration * zoom}px`
                    }}
                    onMouseDown={(e) => handleAudioMouseDown(e, clip.id, clip.id, false)}
                  >
                    <div className="text-xs text-white p-1 truncate pointer-events-none">Vocal</div>

                    <div
                      className="absolute right-0 top-0 bottom-0 w-2 bg-white/30 hover:bg-white/50 cursor-ew-resize"
                      onMouseDown={(e) => handleAudioMouseDown(e, clip.id, clip.id, true)}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
