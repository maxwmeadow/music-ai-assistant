/**
 * Hum2Melody API Client
 *
 * Handles all communication with the backend for humming detection and tuning.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ============================================================
// TypeScript Interfaces
// ============================================================

export interface DetectionParams {
  onsetHigh?: number;
  onsetLow?: number;
  offsetHigh?: number;
  offsetLow?: number;
  minConfidence?: number;
}

export interface WaveformData {
  samples: number[];
  sample_rate: number;
  duration: number;
  original_length: number;
  downsampled_length: number;
}

export interface NoteSegment {
  start: number;
  end: number;
  duration: number;
  pitch: number;
  confidence: number;
  note_name: string;
}

export interface OnsetMarker {
  time: number;
  confidence: number;
}

export interface OffsetMarker {
  time: number;
  confidence: number;
}

export interface VisualizationData {
  segments: NoteSegment[];
  onsets: OnsetMarker[];
  offsets: OffsetMarker[];
  waveform: WaveformData;
  parameters: DetectionParams;
}

export interface UploadResponse {
  status: string;
  ir: any;  // IR object
  session_id: string;
  audio_id: number | null;
  metadata: {
    duration: number;
    num_notes: number;
    instrument: string;
    parameters: DetectionParams;
  };
  visualization?: VisualizationData;
}

export interface ReprocessResponse {
  status: string;
  ir: any;
  session_id: string;
  visualization: VisualizationData;
  metadata: {
    num_notes: number;
    duration: number;
  };
}

// ============================================================
// API Functions
// ============================================================

/**
 * Upload audio with visualization data
 */
export async function uploadWithVisualization(
  audioBlob: Blob,
  params: DetectionParams = {},
  options: {
    instrument?: string;
    saveTrainingData?: boolean;
    returnVisualization?: boolean;
  } = {}
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');
  formData.append('instrument', options.instrument || 'piano/grand_piano_k');
  formData.append('save_training_data', String(options.saveTrainingData !== false));
  // Default to false in production for better performance
  formData.append('return_visualization', String(options.returnVisualization === true));

  // Add detection parameters
  formData.append('onset_high', String(params.onsetHigh ?? 0.30));
  formData.append('onset_low', String(params.onsetLow ?? 0.10));
  formData.append('offset_high', String(params.offsetHigh ?? 0.30));
  formData.append('offset_low', String(params.offsetLow ?? 0.10));
  formData.append('min_confidence', String(params.minConfidence ?? 0.25));

  const response = await fetch(`${API_BASE}/hum2melody`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Reprocess audio with user-provided onset/offset markers
 */
export async function reprocessSegments(
  sessionId: string,
  params: DetectionParams,
  onsets: OnsetMarker[],
  offsets: OffsetMarker[]
): Promise<ReprocessResponse> {
  const formData = new FormData();
  formData.append('session_id', sessionId);

  // Send onset/offset markers (required)
  const onsetTimes = onsets.map(o => o.time);
  const offsetTimes = offsets.map(o => o.time);

  formData.append('manual_onsets', JSON.stringify(onsetTimes));
  formData.append('manual_offsets', JSON.stringify(offsetTimes));

  console.log('[API] Reprocessing with', onsets.length, 'onsets and', offsets.length, 'offsets');

  const response = await fetch(`${API_BASE}/hum2melody/reprocess`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Reprocess failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get segments for a session
 */
export async function getSegments(sessionId: string): Promise<{
  status: string;
  session_id: string;
  visualization: VisualizationData;
}> {
  const response = await fetch(`${API_BASE}/hum2melody/segments/${sessionId}`);

  if (!response.ok) {
    throw new Error(`Get segments failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Delete a session
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/hum2melody/session/${sessionId}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    throw new Error(`Delete session failed: ${response.statusText}`);
  }
}

/**
 * Send IR to runner for compilation
 */
export async function compileIR(ir: any): Promise<{ dsl: string }> {
  const response = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ ir })
  });

  if (!response.ok) {
    throw new Error(`Compile failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================
// Preset Configurations
// ============================================================

export const DETECTION_PRESETS = {
  sensitive: {
    onsetHigh: 0.12,   // Very sensitive for quiet humming
    onsetLow: 0.08,
    offsetHigh: 0.08,  // Very long sustained notes
    offsetLow: 0.06,
    minConfidence: 0.15
  },
  balanced: {
    onsetHigh: 0.15,   // Balanced for typical humming
    onsetLow: 0.10,
    offsetHigh: 0.12,  // Good sustained note length
    offsetLow: 0.10,
    minConfidence: 0.20
  },
  precise: {
    onsetHigh: 0.25,   // Only strong notes
    onsetLow: 0.15,
    offsetHigh: 0.20,  // Moderate note length
    offsetLow: 0.15,
    minConfidence: 0.30
  }
} as const;

export type PresetName = keyof typeof DETECTION_PRESETS;
