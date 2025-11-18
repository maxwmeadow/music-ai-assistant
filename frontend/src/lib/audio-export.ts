/**
 * Audio Export functionality
 * Records Tone.js playback to WAV file
 */

import * as Tone from 'tone';

export interface AudioExportOptions {
  onProgress?: (progress: number) => void;
  onComplete?: (blob: Blob) => void;
  onError?: (error: Error) => void;
}

/**
 * Export current project to WAV audio file
 * This function renders the audio by playing it and recording with Tone.Recorder
 *
 * @param executableCode - The compiled Tone.js code to execute
 * @param duration - Duration in seconds to record
 * @param filename - Optional filename (without extension)
 * @param options - Export options (callbacks)
 */
export async function exportAudio(
  executableCode: string,
  duration: number,
  filename?: string,
  options?: AudioExportOptions
): Promise<Blob> {
  return new Promise(async (resolve, reject) => {
    let recorder: Tone.Recorder | null = null;

    try {
      console.log(`[Audio Export] Starting export (duration: ${duration}s)`);

      // Ensure audio context is running
      if (Tone.getContext().state !== 'running') {
        await Tone.start();
      }

      // Stop any existing transport
      Tone.getTransport().stop();
      Tone.getTransport().cancel();

      // Create recorder connected to master output
      recorder = new Tone.Recorder();
      Tone.getDestination().connect(recorder);

      // Start recording
      recorder.start();
      console.log('[Audio Export] Recording started');

      // Execute the code to set up instruments and schedule notes
      try {
        // Create a safe execution environment
        const func = new Function('Tone', executableCode);
        func(Tone);
      } catch (error) {
        throw new Error(`Failed to execute audio code: ${error}`);
      }

      // Start transport
      Tone.getTransport().start();

      // Monitor progress
      const startTime = Date.now();
      const progressInterval = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        const progress = Math.min(100, (elapsed / duration) * 100);
        if (options?.onProgress) {
          options.onProgress(Math.round(progress));
        }
      }, 100);

      // Wait for duration
      await new Promise(res => setTimeout(res, duration * 1000 + 500)); // Add 500ms buffer

      // Stop recording
      const recording = await recorder.stop();
      clearInterval(progressInterval);

      // Stop transport and cleanup
      Tone.getTransport().stop();
      Tone.getTransport().cancel();
      recorder.dispose();

      console.log(`[Audio Export] Recording complete (${recording.size} bytes)`);

      // Call completion callback
      if (options?.onComplete) {
        options.onComplete(recording);
      }

      resolve(recording);

    } catch (error) {
      console.error('[Audio Export] Export failed:', error);

      // Cleanup on error
      if (recorder) {
        try {
          Tone.getTransport().stop();
          Tone.getTransport().cancel();
          recorder.dispose();
        } catch (cleanupError) {
          console.error('[Audio Export] Cleanup error:', cleanupError);
        }
      }

      const exportError = error instanceof Error
        ? error
        : new Error('Failed to export audio');

      if (options?.onError) {
        options.onError(exportError);
      }

      reject(exportError);
    }
  });
}

/**
 * Download audio blob as WAV file
 * @param blob - Audio blob to download
 * @param filename - Filename (without extension)
 */
export function downloadAudio(blob: Blob, filename: string = 'project'): void {
  try {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.wav`;
    document.body.appendChild(link);
    link.click();

    // Cleanup
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`[Audio Export] Downloaded: ${filename}.wav`);
  } catch (error) {
    console.error('[Audio Export] Download failed:', error);
    throw new Error('Failed to download audio file');
  }
}

/**
 * Export and download audio in one function
 * @param executableCode - Compiled Tone.js code
 * @param duration - Duration to record
 * @param filename - Optional filename
 * @param options - Export options
 */
export async function exportAndDownloadAudio(
  executableCode: string,
  duration: number,
  filename?: string,
  options?: AudioExportOptions
): Promise<void> {
  try {
    const blob = await exportAudio(executableCode, duration, filename, options);
    downloadAudio(blob, filename || 'project');
  } catch (error) {
    console.error('[Audio Export] Export and download failed:', error);
    throw error;
  }
}

/**
 * Calculate duration from DSL code
 * Parses the DSL to find the last note/event time
 * @param dslCode - DSL code string
 * @returns Estimated duration in seconds
 */
export function calculateDurationFromDSL(dslCode: string): number {
  try {
    // Find all note() and chord() calls with timing
    const noteRegex = /note\([^,]+,\s*([0-9.]+)/g;
    const chordRegex = /chord\([^,]+,\s*([0-9.]+)/g;

    let maxTime = 0;
    let match;

    // Check note() calls
    while ((match = noteRegex.exec(dslCode)) !== null) {
      const duration = parseFloat(match[1]);
      if (duration > maxTime) {
        maxTime = duration;
      }
    }

    // Check chord() calls
    while ((match = chordRegex.exec(dslCode)) !== null) {
      const duration = parseFloat(match[1]);
      if (duration > maxTime) {
        maxTime = duration;
      }
    }

    // Add 2 seconds buffer for release/reverb tails
    return maxTime > 0 ? maxTime + 2 : 10; // Default 10s if no notes found
  } catch (error) {
    console.error('[Audio Export] Failed to calculate duration:', error);
    return 10; // Default fallback
  }
}
