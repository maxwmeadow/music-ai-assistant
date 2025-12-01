'use client';

import React from 'react';
import { DetectionParams, DETECTION_PRESETS, PresetName } from '@/lib/hum2melody-api';

interface ParameterControlPanelProps {
  params: DetectionParams;
  onChange: (params: DetectionParams) => void;
  onReprocess: () => void;
  isProcessing?: boolean;
}

export default function ParameterControlPanel({
  params,
  onChange,
  onReprocess,
  isProcessing = false
}: ParameterControlPanelProps) {
  const updateParam = (key: keyof DetectionParams, value: number) => {
    onChange({ ...params, [key]: value });
  };

  const applyPreset = (presetName: PresetName) => {
    onChange(DETECTION_PRESETS[presetName]);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Detection Parameters</h3>
        <div className="flex gap-2">
          <PresetButton
            onClick={() => applyPreset('sensitive')}
            label="Sensitive"
            description="Catches more notes"
          />
          <PresetButton
            onClick={() => applyPreset('balanced')}
            label="Balanced"
            description="Recommended"
          />
          <PresetButton
            onClick={() => applyPreset('precise')}
            label="Precise"
            description="Only confident"
          />
        </div>
      </div>

      <div className="space-y-3">
        {/* Onset Detection */}
        <div className="space-y-2">
          <div className="text-sm text-gray-300 font-medium">Onset Detection</div>
          <Slider
            label="High Threshold"
            value={params.onsetHigh ?? 0.30}
            onChange={(v) => updateParam('onsetHigh', v)}
            min={0.10}
            max={0.50}
            step={0.05}
            tooltip="Lower = detect more note starts"
          />
          <Slider
            label="Low Threshold"
            value={params.onsetLow ?? 0.10}
            onChange={(v) => updateParam('onsetLow', v)}
            min={0.05}
            max={0.30}
            step={0.05}
            tooltip="Hysteresis threshold"
          />
        </div>

        {/* Offset Detection */}
        <div className="space-y-2">
          <div className="text-sm text-gray-300 font-medium">Offset Detection</div>
          <Slider
            label="High Threshold"
            value={params.offsetHigh ?? 0.30}
            onChange={(v) => updateParam('offsetHigh', v)}
            min={0.10}
            max={0.50}
            step={0.05}
            tooltip="Lower = longer notes"
          />
          <Slider
            label="Low Threshold"
            value={params.offsetLow ?? 0.10}
            onChange={(v) => updateParam('offsetLow', v)}
            min={0.05}
            max={0.30}
            step={0.05}
            tooltip="Hysteresis threshold"
          />
        </div>

        {/* Confidence Filter */}
        <div className="space-y-2">
          <div className="text-sm text-gray-300 font-medium">Confidence Filter</div>
          <Slider
            label="Min Confidence"
            value={params.minConfidence ?? 0.25}
            onChange={(v) => updateParam('minConfidence', v)}
            min={0.10}
            max={0.50}
            step={0.05}
            tooltip="Lower = keep more notes (some may be wrong)"
          />
        </div>
      </div>

      {/* Reprocess Button */}
      <button
        onClick={onReprocess}
        disabled={isProcessing}
        className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600
                   text-white font-medium rounded transition-colors"
      >
        {isProcessing ? 'Processing...' : 'Re-process with new parameters'}
      </button>

      {/* Help Text */}
      <div className="text-xs text-gray-500 space-y-1">
        <div><strong>Missing notes?</strong> Lower onset high threshold</div>
        <div><strong>Notes too short?</strong> Lower offset high threshold</div>
        <div><strong>Too many false notes?</strong> Raise confidence threshold</div>
      </div>
    </div>
  );
}

// ============================================================
// Slider Component
// ============================================================

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  tooltip?: string;
}

function Slider({ label, value, onChange, min, max, step, tooltip }: SliderProps) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-xs text-gray-400 w-32" title={tooltip}>
        {label}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-blue-500
                   [&::-webkit-slider-thumb]:cursor-pointer"
      />
      <span className="text-xs text-gray-300 w-12 text-right font-mono">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

// ============================================================
// Preset Button Component
// ============================================================

interface PresetButtonProps {
  onClick: () => void;
  label: string;
  description: string;
}

function PresetButton({ onClick, label, description }: PresetButtonProps) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded
                 transition-colors"
      title={description}
    >
      {label}
    </button>
  );
}
