'use client';

import React, { useState } from 'react';
import WaveformVisualizer from './WaveformVisualizer';
import {
  VisualizationData,
  NoteSegment,
  reprocessSegments
} from '@/lib/hum2melody-api';

interface DetectionTunerProps {
  sessionId: string;
  initialVisualization: VisualizationData;
  initialIR: any;
  onApply: (ir: any, segments: NoteSegment[]) => void;
  onCancel: () => void;
}

export default function DetectionTuner({
  sessionId,
  initialVisualization,
  initialIR,
  onApply,
  onCancel
}: DetectionTunerProps) {
  const [visualization, setVisualization] = useState(initialVisualization);
  const [currentIR, setCurrentIR] = useState(initialIR);
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reprocess with current markers to get pitch predictions
  const handleReprocess = async () => {
    setIsProcessing(true);
    setError(null);

    try {
      const { onsets, offsets } = visualization;

      console.log('[DetectionTuner] Reprocessing with markers:', {
        onsets: onsets.length,
        offsets: offsets.length
      });

      // Send current markers to backend for pitch prediction
      const response = await reprocessSegments(
        sessionId,
        {}, // No parameters needed - just use the markers
        onsets,
        offsets
      );

      console.log('[DetectionTuner] Got', response.visualization.segments.length, 'segments');

      // Update with new segments (pitch predictions)
      setVisualization({
        ...visualization,
        segments: response.visualization.segments
      });
      setCurrentIR(response.ir);
      setSelectedSegmentIndex(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Reprocessing failed');
      console.error('Reprocess error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Apply and close - reconstruct IR from current visualization segments
  const handleApply = () => {
    // Reconstruct IR from current visualization segments
    // This ensures we use the segments shown in the UI, not the stale model inference
    const reconstructedIR = {
      ...currentIR,
      tracks: currentIR.tracks.map((track: any) => {
        // Only update tracks with notes (melody), not drum samples
        if (track.notes !== undefined && track.notes !== null) {
          const notes = visualization.segments.map((seg: NoteSegment) => ({
            pitch: seg.pitch,
            start: seg.start,
            duration: seg.duration,
            velocity: seg.confidence // Use confidence as velocity
          }));

          return {
            ...track,
            notes
          };
        }
        return track;
      })
    };

    console.log(`[DetectionTuner] Applying IR with ${visualization.segments.length} notes from visualization`);
    onApply(reconstructedIR, visualization.segments);
  };

  const { segments, waveform, onsets, offsets } = visualization;

  return (
    <div className="fixed inset-0 z-50 bg-black bg-opacity-75 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-lg shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-auto">
        {/* Header */}
        <div className="sticky top-0 bg-gray-900 border-b border-gray-700 p-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Fine-Tune Detection</h2>
            <p className="text-sm text-gray-400">
              Adjust parameters to improve note detection accuracy
            </p>
          </div>
          <button
            onClick={onCancel}
            className="text-gray-400 hover:text-white transition-colors"
            title="Close"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Main Content */}
        <div className="p-4 space-y-4">
          {/* Error Message */}
          {error && (
            <div className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <StatCard
              label="Detected Notes"
              value={segments.length}
              color="blue"
            />
            <StatCard
              label="Duration"
              value={`${waveform.duration.toFixed(1)}s`}
              color="green"
            />
            <StatCard
              label="Avg Confidence"
              value={segments.length > 0
                ? `${(segments.reduce((sum, s) => sum + s.confidence, 0) / segments.length * 100).toFixed(0)}%`
                : 'N/A'}
              color="purple"
            />
          </div>

          {/* Waveform Visualizer */}
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-white">Waveform & Segments</h3>
              <div className="text-xs text-gray-400">
                Edit markers, then click Re-process to get pitch predictions
              </div>
            </div>
            <WaveformVisualizer
              waveformData={waveform}
              onsets={onsets}
              offsets={offsets}
              segments={segments}
              selectedSegmentIndex={selectedSegmentIndex}
              onSegmentClick={setSelectedSegmentIndex}
              onMarkersChanged={(newOnsets, newOffsets) => {
                console.log('[DetectionTuner] Markers changed:', {
                  onsets: newOnsets.length,
                  offsets: newOffsets.length
                });

                // Create preview segments by pairing onsets with offsets
                const previewSegments: NoteSegment[] = [];
                const sortedOnsets = [...newOnsets].sort((a, b) => a.time - b.time);
                const sortedOffsets = [...newOffsets].sort((a, b) => a.time - b.time);
                const usedOffsets = new Set<number>();

                for (const onset of sortedOnsets) {
                  // Find next offset after this onset that hasn't been used
                  const matchingOffset = sortedOffsets.find(
                    off => off.time > onset.time && !usedOffsets.has(off.time)
                  );

                  if (matchingOffset) {
                    previewSegments.push({
                      start: onset.time,
                      end: matchingOffset.time,
                      duration: matchingOffset.time - onset.time,
                      pitch: 60, // Placeholder pitch for preview
                      confidence: 0.5,
                      note_name: 'Preview'
                    });
                    usedOffsets.add(matchingOffset.time);
                  }
                }

                console.log('[DetectionTuner] Created', previewSegments.length, 'preview segments');

                // Update visualization with preview segments and new markers
                setVisualization({
                  ...visualization,
                  segments: previewSegments,
                  onsets: newOnsets,
                  offsets: newOffsets
                });
              }}
              height={250}
              editable={true}
            />
          </div>

          {/* Selected Segment Info */}
          {selectedSegmentIndex !== null && segments[selectedSegmentIndex] && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-white mb-2">Selected Segment</h3>
              <SegmentInfo segment={segments[selectedSegmentIndex]} />
            </div>
          )}

          {/* Segment List */}
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-white">
                Detected Notes ({segments.length})
              </h3>
              <button
                onClick={handleReprocess}
                disabled={isProcessing}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded transition-colors text-sm"
              >
                {isProcessing ? 'Processing...' : 'Re-process with Current Markers'}
              </button>
            </div>
            <div className="max-h-96 overflow-y-auto space-y-1">
              {segments.map((segment, index) => (
                <SegmentListItem
                  key={index}
                  segment={segment}
                  index={index}
                  isSelected={index === selectedSegmentIndex}
                  onClick={() => setSelectedSegmentIndex(index)}
                />
              ))}
              {segments.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  No notes yet. Edit markers above, then click Re-process.
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 justify-end pt-4 border-t border-gray-700">
            <button
              onClick={onCancel}
              className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleApply}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded transition-colors"
            >
              Apply & Continue
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Sub-components
// ============================================================

interface StatCardProps {
  label: string;
  value: string | number;
  color: 'blue' | 'green' | 'purple';
}

function StatCard({ label, value, color }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-900 border-blue-700',
    green: 'bg-green-900 border-green-700',
    purple: 'bg-purple-900 border-purple-700'
  };

  return (
    <div className={`${colorClasses[color]} border rounded-lg p-3`}>
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-2xl font-bold text-white mt-1">{value}</div>
    </div>
  );
}

interface SegmentInfoProps {
  segment: NoteSegment;
}

function SegmentInfo({ segment }: SegmentInfoProps) {
  return (
    <div className="grid grid-cols-2 gap-2 text-sm">
      <div>
        <span className="text-gray-400">Note:</span>
        <span className="text-white ml-2 font-mono">{segment.note_name}</span>
      </div>
      <div>
        <span className="text-gray-400">MIDI:</span>
        <span className="text-white ml-2 font-mono">{segment.pitch}</span>
      </div>
      <div>
        <span className="text-gray-400">Start:</span>
        <span className="text-white ml-2 font-mono">{segment.start.toFixed(2)}s</span>
      </div>
      <div>
        <span className="text-gray-400">End:</span>
        <span className="text-white ml-2 font-mono">{segment.end.toFixed(2)}s</span>
      </div>
      <div>
        <span className="text-gray-400">Duration:</span>
        <span className="text-white ml-2 font-mono">{segment.duration.toFixed(2)}s</span>
      </div>
      <div>
        <span className="text-gray-400">Confidence:</span>
        <span className="text-white ml-2 font-mono">{(segment.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

interface SegmentListItemProps {
  segment: NoteSegment;
  index: number;
  isSelected: boolean;
  onClick: () => void;
}

function SegmentListItem({ segment, index, isSelected, onClick }: SegmentListItemProps) {
  return (
    <div
      onClick={onClick}
      className={`
        px-3 py-2 rounded cursor-pointer transition-colors text-sm
        ${isSelected
          ? 'bg-blue-600 text-white'
          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        }
      `}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold">{segment.note_name}</span>
          <span className="text-xs opacity-70">
            {segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="text-xs opacity-70">
            {(segment.confidence * 100).toFixed(0)}%
          </div>
          <div
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor: `hsl(${120 + segment.confidence * 60}, 70%, 50%)`
            }}
          />
        </div>
      </div>
    </div>
  );
}
