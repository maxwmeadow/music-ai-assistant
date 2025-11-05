'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { WaveformData, NoteSegment, OnsetMarker, OffsetMarker } from '@/lib/hum2melody-api';

interface WaveformVisualizerProps {
  waveformData: WaveformData;
  segments: NoteSegment[];
  onsets?: OnsetMarker[];
  offsets?: OffsetMarker[];
  selectedSegmentIndex?: number | null;
  onSegmentClick?: (index: number) => void;
  onMarkersChanged?: (onsets: OnsetMarker[], offsets: OffsetMarker[]) => void;
  height?: number;
  editable?: boolean;
}

type DragState =
  | { type: 'marker', markerType: 'onset' | 'offset', index: number }
  | { type: 'creating', startTime: number }
  | null;

export default function WaveformVisualizer({
  waveformData,
  segments,
  onsets = [],
  offsets = [],
  selectedSegmentIndex = null,
  onSegmentClick,
  onMarkersChanged,
  height = 200,
  editable = true
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasWidth, setCanvasWidth] = useState(800);

  // Simplified editing state
  const [hoveredMarkerIndex, setHoveredMarkerIndex] = useState<{type: 'onset' | 'offset', index: number} | null>(null);
  const [dragState, setDragState] = useState<DragState>(null);
  const [localOnsets, setLocalOnsets] = useState<OnsetMarker[]>(onsets);
  const [localOffsets, setLocalOffsets] = useState<OffsetMarker[]>(offsets);
  const [previewPair, setPreviewPair] = useState<{start: number, end: number} | null>(null);

  // Update local markers when props change
  useEffect(() => {
    setLocalOnsets(onsets);
    setLocalOffsets(offsets);
  }, [onsets, offsets]);

  // Responsive canvas width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setCanvasWidth(containerRef.current.clientWidth);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Helper: Convert click X position to time
  const xToTime = useCallback((x: number, canvas: HTMLCanvasElement) => {
    return (x / canvas.width) * waveformData.duration;
  }, [waveformData.duration]);

  // Helper: Convert time to X position
  const timeToX = useCallback((time: number, canvas: HTMLCanvasElement) => {
    return (time / waveformData.duration) * canvas.width;
  }, [waveformData.duration]);

  // Find marker near click position
  const findMarkerAtPosition = useCallback((x: number, canvas: HTMLCanvasElement): {type: 'onset' | 'offset', index: number} | null => {
    const time = xToTime(x, canvas);
    const tolerance = 0.1; // seconds

    // Check onsets
    for (let i = 0; i < localOnsets.length; i++) {
      if (Math.abs(localOnsets[i].time - time) < tolerance) {
        return { type: 'onset', index: i };
      }
    }

    // Check offsets
    for (let i = 0; i < localOffsets.length; i++) {
      if (Math.abs(localOffsets[i].time - time) < tolerance) {
        return { type: 'offset', index: i };
      }
    }

    return null;
  }, [localOnsets, localOffsets, xToTime]);

  // Draw waveform and markers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { samples, duration } = waveformData;
    const width = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, h);

    // Draw grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;

    // Horizontal center line
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(width, h / 2);
    ctx.stroke();

    // Vertical time markers (every second)
    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    const secondsToShow = Math.ceil(duration);
    for (let i = 0; i <= secondsToShow; i++) {
      const x = (i / duration) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(`${i}s`, x + 2, 12);
    }

    // Draw segments (as colored bars behind waveform)
    segments.forEach((segment, index) => {
      const startX = (segment.start / duration) * width;
      const endX = (segment.end / duration) * width;
      const segmentWidth = endX - startX;

      // Color based on selection
      const isSelected = index === selectedSegmentIndex;
      const alpha = isSelected ? 0.4 : 0.25;

      // Color by confidence
      const hue = 120 + (segment.confidence * 60); // Green to yellow
      ctx.fillStyle = `hsla(${hue}, 70%, 50%, ${alpha})`;
      ctx.fillRect(startX, 0, segmentWidth, h);

      // Border
      if (isSelected) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, 0, segmentWidth, h);
      }

      // Note label
      ctx.fillStyle = isSelected ? '#fff' : '#aaa';
      ctx.font = '12px monospace';
      ctx.fillText(
        segment.note_name,
        startX + 2,
        h - 5
      );
    });

    // Draw onset markers (red lines)
    localOnsets.forEach((onset, index) => {
      const x = (onset.time / duration) * width;
      const isHovered = hoveredMarkerIndex?.type === 'onset' && hoveredMarkerIndex.index === index;
      const isDragging = dragState?.type === 'marker' && dragState.markerType === 'onset' && dragState.index === index;

      ctx.strokeStyle = isDragging ? '#ffff00' : (isHovered ? '#ff9999' : '#ff6b6b');
      ctx.lineWidth = isDragging ? 4 : (isHovered ? 3 : 2);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();

      // Draw handle at top for easier grabbing
      if (editable) {
        ctx.fillStyle = isDragging ? '#ffff00' : (isHovered ? '#ff9999' : '#ff6b6b');
        ctx.fillRect(x - 4, 0, 8, 12);
      }
    });

    // Draw offset markers (cyan lines)
    localOffsets.forEach((offset, index) => {
      const x = (offset.time / duration) * width;
      const isHovered = hoveredMarkerIndex?.type === 'offset' && hoveredMarkerIndex.index === index;
      const isDragging = dragState?.type === 'marker' && dragState.markerType === 'offset' && dragState.index === index;

      ctx.strokeStyle = isDragging ? '#ffff00' : (isHovered ? '#99ffff' : '#4ecdc4');
      ctx.lineWidth = isDragging ? 4 : (isHovered ? 3 : 2);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();

      // Draw handle at bottom for easier grabbing
      if (editable) {
        ctx.fillStyle = isDragging ? '#ffff00' : (isHovered ? '#99ffff' : '#4ecdc4');
        ctx.fillRect(x - 4, h - 12, 8, 12);
      }
    });

    // Draw waveform as filled blob (envelope style)
    const centerY = h / 2;
    const amplitude = h * 0.4;

    // Draw filled envelope (positive)
    ctx.fillStyle = 'rgba(0, 170, 255, 0.3)';
    ctx.beginPath();
    ctx.moveTo(0, centerY);

    samples.forEach((sample, i) => {
      const x = (i / samples.length) * width;
      const y = centerY - Math.abs(sample) * amplitude;
      ctx.lineTo(x, y);
    });

    ctx.lineTo(width, centerY);
    ctx.closePath();
    ctx.fill();

    // Draw outline for definition
    ctx.strokeStyle = '#00aaff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    samples.forEach((sample, i) => {
      const x = (i / samples.length) * width;
      const y = centerY - Math.abs(sample) * amplitude;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw preview pair if creating
    if (previewPair) {
      const startX = (Math.min(previewPair.start, previewPair.end) / duration) * width;
      const endX = (Math.max(previewPair.start, previewPair.end) / duration) * width;

      // Semi-transparent yellow preview
      ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
      ctx.fillRect(startX, 0, endX - startX, h);

      // Preview markers
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(startX, 0);
      ctx.lineTo(startX, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(endX, 0);
      ctx.lineTo(endX, h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

  }, [waveformData, segments, localOnsets, localOffsets, selectedSegmentIndex, canvasWidth, height, hoveredMarkerIndex, dragState, previewPair, editable]);

  // Mouse move handler
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editable) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = xToTime(x, canvas);
    const clampedTime = Math.max(0, Math.min(time, waveformData.duration));

    if (dragState) {
      if (dragState.type === 'marker') {
        // Dragging existing marker - update live
        if (dragState.markerType === 'onset') {
          const newOnsets = [...localOnsets];
          newOnsets[dragState.index] = { ...newOnsets[dragState.index], time: clampedTime };
          setLocalOnsets(newOnsets);

          // Notify parent immediately for live preview update
          if (onMarkersChanged) {
            onMarkersChanged(newOnsets, localOffsets);
          }
        } else {
          const newOffsets = [...localOffsets];
          newOffsets[dragState.index] = { ...newOffsets[dragState.index], time: clampedTime };
          setLocalOffsets(newOffsets);

          // Notify parent immediately for live preview update
          if (onMarkersChanged) {
            onMarkersChanged(localOnsets, newOffsets);
          }
        }
      } else if (dragState.type === 'creating') {
        // Creating new pair - show preview
        setPreviewPair({ start: dragState.startTime, end: clampedTime });
      }
    } else {
      // Check for hover
      const marker = findMarkerAtPosition(x, canvas);
      setHoveredMarkerIndex(marker);
    }
  }, [editable, dragState, localOnsets, localOffsets, xToTime, findMarkerAtPosition, waveformData.duration, onMarkersChanged]);

  // Mouse down handler
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editable) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = xToTime(x, canvas);

    // Check if clicking on existing marker
    const marker = findMarkerAtPosition(x, canvas);

    if (marker) {
      // Start dragging existing marker
      setDragState({ type: 'marker', markerType: marker.type, index: marker.index });
    } else {
      // Start creating new onset/offset pair
      setDragState({ type: 'creating', startTime: time });
    }
  }, [editable, findMarkerAtPosition, xToTime]);

  // Mouse up handler
  const handleMouseUp = useCallback(() => {
    if (!dragState) return;

    if (dragState.type === 'marker') {
      // Finished dragging marker - notify parent
      if (onMarkersChanged) {
        onMarkersChanged(localOnsets, localOffsets);
      }
    } else if (dragState.type === 'creating' && previewPair) {
      // Finished creating new pair - add onset and offset
      const newOnset: OnsetMarker = { time: Math.min(previewPair.start, previewPair.end), confidence: 1.0 };
      const newOffset: OffsetMarker = { time: Math.max(previewPair.start, previewPair.end), confidence: 1.0 };

      const newOnsets = [...localOnsets, newOnset].sort((a, b) => a.time - b.time);
      const newOffsets = [...localOffsets, newOffset].sort((a, b) => a.time - b.time);

      setLocalOnsets(newOnsets);
      setLocalOffsets(newOffsets);

      if (onMarkersChanged) {
        onMarkersChanged(newOnsets, newOffsets);
      }
    }

    setDragState(null);
    setPreviewPair(null);
  }, [dragState, previewPair, localOnsets, localOffsets, onMarkersChanged]);

  // Key handler for deleting markers
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!editable || !hoveredMarkerIndex) return;

    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (hoveredMarkerIndex.type === 'onset') {
        const newOnsets = localOnsets.filter((_, i) => i !== hoveredMarkerIndex.index);
        setLocalOnsets(newOnsets);
        if (onMarkersChanged) {
          onMarkersChanged(newOnsets, localOffsets);
        }
      } else {
        const newOffsets = localOffsets.filter((_, i) => i !== hoveredMarkerIndex.index);
        setLocalOffsets(newOffsets);
        if (onMarkersChanged) {
          onMarkersChanged(localOnsets, newOffsets);
        }
      }
      setHoveredMarkerIndex(null);
    }
  }, [editable, hoveredMarkerIndex, localOnsets, localOffsets, onMarkersChanged]);

  // Attach keyboard listener
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div ref={containerRef} className="w-full space-y-2">
      {/* Instructions */}
      {editable && (
        <div className="text-xs text-gray-400">
          <strong>Click and drag</strong> on waveform to create onset/offset pairs •
          <strong> Drag markers</strong> to adjust timing •
          <strong> Hover + Delete</strong> to remove markers
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`w-full rounded border border-gray-700 ${
          dragState?.type === 'creating' ? 'cursor-crosshair' :
          dragState?.type === 'marker' ? 'cursor-grabbing' :
          hoveredMarkerIndex ? 'cursor-grab' :
          'cursor-crosshair'
        }`}
        style={{ imageRendering: 'pixelated' }}
      />

      {/* Legend */}
      <div className="flex gap-4 text-xs text-gray-400 flex-wrap">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-blue-500"></div>
          <span>Waveform</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 opacity-40"></div>
          <span>Detected Segments</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-1 bg-red-500"></div>
          <span>Onsets (Start)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-1 bg-cyan-500"></div>
          <span>Offsets (End)</span>
        </div>
        {editable && (
          <span className="text-yellow-400">
            Hover + Delete to remove markers
          </span>
        )}
      </div>
    </div>
  );
}
