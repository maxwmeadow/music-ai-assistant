'use client';

import React, { useRef, useEffect, useState } from 'react';

interface DrumHit {
  time: number;
  drum_type: string;
  confidence: number;
  probabilities?: {
    kick: number;
    snare: number;
    hihat: number;
  };
}

interface WaveformData {
  data: number[];
  sample_rate: number;
  duration: number;
}

interface VisualizationData {
  waveform: WaveformData;
  drum_hits: DrumHit[];
  num_hits: number;
}

interface DrumOnsetVisualizerProps {
  visualization: VisualizationData;
  onClose: () => void;
}

const DrumOnsetVisualizer: React.FC<DrumOnsetVisualizerProps> = ({
  visualization,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredOnset, setHoveredOnset] = useState<number | null>(null);

  // Draw waveform and onset markers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { waveform, drum_hits } = visualization;
    const { data, duration } = waveform;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Draw waveform
    ctx.strokeStyle = '#4a5568';
    ctx.lineWidth = 1;
    ctx.beginPath();

    const centerY = height / 2;
    const amp = height * 0.35; // Use 70% of height for waveform

    data.forEach((value, i) => {
      const x = (i / data.length) * width;
      const y = centerY + value * amp;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw onset markers and probabilities
    const drumColors = {
      kick: '#ef4444', // red
      snare: '#3b82f6', // blue
      hihat: '#10b981', // green
    };

    drum_hits.forEach((hit, index) => {
      const x = (hit.time / duration) * width;
      const isHovered = index === hoveredOnset;

      // Draw vertical line at onset
      ctx.strokeStyle = drumColors[hit.drum_type as keyof typeof drumColors] || '#888';
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.setLineDash(isHovered ? [] : [5, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw probabilities if available
      if (hit.probabilities) {
        const barHeight = 60;
        const barWidth = 40;
        const barX = Math.min(Math.max(x - barWidth / 2, 5), width - barWidth - 5);
        const barY = 10;

        // Draw background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(barX, barY, barWidth, barHeight);

        // Draw probability bars
        const probs = [
          { label: 'K', value: hit.probabilities.kick, color: drumColors.kick },
          { label: 'S', value: hit.probabilities.snare, color: drumColors.snare },
          { label: 'H', value: hit.probabilities.hihat, color: drumColors.hihat },
        ];

        const barSpacing = 3;
        const innerWidth = barWidth - 10;
        const maxBarHeight = barHeight - 25;

        probs.forEach((prob, i) => {
          const segmentHeight = maxBarHeight / 3;
          const segmentY = barY + 5 + i * segmentHeight;

          // Draw label
          ctx.fillStyle = '#ffffff';
          ctx.font = '10px monospace';
          ctx.fillText(prob.label, barX + 3, segmentY + 12);

          // Draw bar
          const barFillWidth = (prob.value * (innerWidth - 15));
          ctx.fillStyle = prob.color;
          ctx.fillRect(barX + 18, segmentY + 4, barFillWidth, segmentHeight - barSpacing);

          // Draw percentage
          ctx.fillStyle = '#ffffff';
          ctx.font = '9px monospace';
          const pctText = `${(prob.value * 100).toFixed(0)}%`;
          ctx.fillText(pctText, barX + 20 + barFillWidth + 2, segmentY + 12);
        });

        // Draw predicted drum type at top
        ctx.fillStyle = drumColors[hit.drum_type as keyof typeof drumColors];
        ctx.font = 'bold 10px monospace';
        const drumLabel = hit.drum_type.toUpperCase();
        ctx.fillText(drumLabel, barX + 3, barY + barHeight - 3);
      }
    });
  }, [visualization, hoveredOnset]);

  // Handle mouse move for hover effects
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;

    const { drum_hits, waveform } = visualization;
    const { duration } = waveform;

    // Find closest onset
    let closestIndex: number | null = null;
    let closestDistance = Infinity;

    drum_hits.forEach((hit, index) => {
      const hitX = (hit.time / duration) * width;
      const distance = Math.abs(x - hitX);

      if (distance < closestDistance && distance < 20) {
        // Within 20px
        closestDistance = distance;
        closestIndex = index;
      }
    });

    setHoveredOnset(closestIndex);
  };

  const handleMouseLeave = () => {
    setHoveredOnset(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75">
      <div className="bg-gray-900 rounded-lg shadow-xl w-[90vw] max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div>
            <h2 className="text-xl font-bold text-white">Drum Onset Visualization</h2>
            <p className="text-sm text-gray-400 mt-1">
              {visualization.num_hits} detections | {visualization.waveform.duration.toFixed(2)}s duration
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors px-3 py-1 rounded"
          >
            ✕ Close
          </button>
        </div>

        {/* Waveform Canvas */}
        <div className="flex-1 p-4 overflow-hidden">
          <canvas
            ref={canvasRef}
            className="w-full h-full cursor-crosshair"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
        </div>

        {/* Legend */}
        <div className="p-4 border-t border-gray-700 bg-gray-800">
          <div className="flex items-center gap-6 text-sm">
            <span className="text-gray-300 font-semibold">Legend:</span>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span className="text-gray-300">Kick</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span className="text-gray-300">Snare</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span className="text-gray-300">Hihat</span>
            </div>
            <span className="text-gray-400 ml-4">Hover over onset markers to highlight</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DrumOnsetVisualizer;