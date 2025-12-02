/**
 * Audio Service - Handles Tone.js audio playback and transport management
 */

interface TransportState {
  seconds: number;
  state: 'started' | 'paused' | 'stopped';
}

export class AudioService {
  private static autoStopTimeout: NodeJS.Timeout | null = null;

  /**
   * Initialize Tone.js and set up transport
   */
  static async initializeTone(trackVolumes: Record<string, number> = {}) {
    const Tone = await import('tone');
    (window as any).Tone = Tone;
    (window as any).__trackVolumes = trackVolumes;
    return Tone;
  }

  /**
   * Reset transport to initial state
   */
  static resetTransport(setCurrentTime?: (time: number) => void) {
    if ((window as any).Tone?.Transport) {
      const Tone = (window as any).Tone;
      if (Tone.Transport.state === 'started') {
        Tone.Transport.stop();
      }
      Tone.Transport.seconds = 0;
      Tone.Transport.cancel();
      if (setCurrentTime) {
        setCurrentTime(0);
      }
    }
  }

  /**
   * Start or resume transport playback
   */
  static startTransport() {
    if ((window as any).Tone?.Transport) {
      (window as any).Tone.Transport.start();
    }
  }

  /**
   * Pause transport (preserves position)
   */
  static pauseTransport(): boolean {
    if ((window as any).Tone?.Transport) {
      const wasPlaying = (window as any).Tone.Transport.state === 'started';
      (window as any).Tone.Transport.pause();
      return wasPlaying;
    }
    return false;
  }

  /**
   * Stop transport (resets to 0)
   */
  static stopTransport() {
    if ((window as any).Tone?.Transport) {
      (window as any).Tone.Transport.stop();
      (window as any).Tone.Transport.seconds = 0;
    }
  }

  /**
   * Get current transport state
   */
  static getTransportState(): TransportState | null {
    if ((window as any).Tone?.Transport) {
      return {
        seconds: (window as any).Tone.Transport.seconds,
        state: (window as any).Tone.Transport.state,
      };
    }
    return null;
  }

  /**
   * Seek to a specific time
   */
  static seek(time: number) {
    if ((window as any).Tone?.Transport) {
      (window as any).Tone.Transport.seconds = time;
    }
  }

  /**
   * Set auto-stop timeout for playback
   */
  static setAutoStop(duration: number, onStop: () => void) {
    console.log('[AudioService] [DEBUG] setAutoStop() called with duration:', duration);
    this.clearAutoStop();
    this.autoStopTimeout = setTimeout(() => {
      console.log('[AudioService] [DEBUG] ===== FRONTEND AUTO-STOP TIMEOUT FIRED =====');
      console.log('[AudioService] [DEBUG] Transport state:', (window as any).Tone?.Transport.state);
      if ((window as any).Tone?.Transport.state === 'started') {
        console.log('[AudioService] [DEBUG] Calling onStop callback');
        onStop();
      } else {
        console.log('[AudioService] [DEBUG] Transport not started, skipping onStop callback');
      }
    }, duration * 1000);
    (window as any).__autoStopTimeout = this.autoStopTimeout;
    console.log('[AudioService] [DEBUG] Auto-stop timeout set (ID:', this.autoStopTimeout, ')');
  }

  /**
   * Clear auto-stop timeout
   */
  static clearAutoStop() {
    console.log('[AudioService] [DEBUG] clearAutoStop() called');
    if (this.autoStopTimeout) {
      console.log('[AudioService] [DEBUG] Clearing autoStopTimeout:', this.autoStopTimeout);
      clearTimeout(this.autoStopTimeout);
      this.autoStopTimeout = null;
    }
    if ((window as any).__autoStopTimeout) {
      console.log('[AudioService] [DEBUG] Clearing window.__autoStopTimeout');
      clearTimeout((window as any).__autoStopTimeout);
      (window as any).__autoStopTimeout = null;
    }
  }

  /**
   * Check if samples are preloaded
   */
  static areSamplesPreloaded(): boolean {
    return !!(window as any).__samplesPreloaded;
  }

  /**
   * Mark samples as preloaded
   */
  static markSamplesPreloaded() {
    (window as any).__samplesPreloaded = true;
  }

  /**
   * Preload audio samples by evaluating code without playback
   */
  static async preloadSamples(
    execCode: string,
    trackVolumes: Record<string, number>,
    onLoadingStateChange: (isLoading: boolean) => void
  ) {
    console.log('[AudioService] [DEBUG] preloadSamples() called');
    onLoadingStateChange(true);
    (window as any).__isPreloading = true;
    console.log('[AudioService] [DEBUG] Set __isPreloading = true');

    try {
      const Tone = await this.initializeTone(trackVolumes);
      console.log('[AudioService] [DEBUG] Tone initialized');

      // Reset transport state
      this.resetTransport();
      console.log('[AudioService] [DEBUG] Transport reset');

      // Evaluate code and intercept Transport.start() to prevent autoplay
      const originalStart = Tone.Transport.start;
      console.log('[AudioService] [DEBUG] Intercepting Transport.start()');
      Tone.Transport.start = () => {
        console.log('[AudioService] [DEBUG] Intercepted Transport.start() called - NOT starting transport');
        return Tone.Transport as any;
      };

      console.log('[AudioService] [DEBUG] Evaluating executable code...');
      eval(execCode);
      console.log('[AudioService] [DEBUG] Executable code evaluated');

      Tone.Transport.start = originalStart;
      console.log('[AudioService] [DEBUG] Restored original Transport.start()');

      // Mark as preloaded and trigger actual loading
      this.markSamplesPreloaded();
      console.log('[AudioService] [DEBUG] Marked samples as preloaded');

      console.log('[AudioService] [DEBUG] Starting transport for actual loading...');
      Tone.Transport.start();
      console.log('[AudioService] [DEBUG] Transport started');

      // Loading state will be cleared when "[Music] Playing..." is logged
    } catch (error) {
      console.error('[AudioService] [DEBUG] Preload error:', error);
      (window as any).__isPreloading = false;
      onLoadingStateChange(false);
      throw error;
    }
  }

  /**
   * Handle preload completion (called when samples are ready)
   */
  static handlePreloadComplete() {
    console.log('[AudioService] [DEBUG] handlePreloadComplete() called');
    console.log('[AudioService] [DEBUG] __isPreloading:', (window as any).__isPreloading);
    if ((window as any).__isPreloading) {
      console.log('[AudioService] [DEBUG] Stopping transport after preload');
      this.stopTransport();
      (window as any).__isPreloading = false;
      console.log('[AudioService] [DEBUG] Set __isPreloading = false');
    }
  }

  /**
   * Check if currently in preload mode
   */
  static isPreloading(): boolean {
    return !!(window as any).__isPreloading;
  }
}
