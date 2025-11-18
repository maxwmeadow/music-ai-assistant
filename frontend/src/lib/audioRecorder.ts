
export type RecorderEvents = {
    onChunk?: (blobPart: Blob) => void;
    onStart?: () => void;
    onStop?: (finalBlob: Blob) => void;
    onError?: (err: Error) => void;
    onStreamActive?: (active: boolean) => void;
    onDurationUpdate?: (duration: number) => void;
    onClippingDetected?: (isClipping: boolean) => void;
};

export class AudioRecorder {
    private mediaRecorder: MediaRecorder | null = null;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private stream: MediaStream | null = null;
    private chunks: Blob[] = [];
    private sourceNode: MediaStreamAudioSourceNode | null = null;
    private startTime: number = 0;
    private durationInterval: NodeJS.Timeout | null = null;
    private streamCheckInterval: NodeJS.Timeout | null = null;
    private clippingCheckInterval: NodeJS.Timeout | null = null;
    private clippingThreshold: number = 0.95; // Detect clipping above 95% of max amplitude

    constructor(private events: RecorderEvents = {}) {}

    public getAnalyser(): AnalyserNode | null {
        return this.analyser;
    }

    /**
     * Check if MediaRecorder API is supported
     */
    public static isSupported(): boolean {
        return typeof MediaRecorder !== 'undefined' && 
               typeof navigator !== 'undefined' && 
               typeof navigator.mediaDevices !== 'undefined' &&
               typeof navigator.mediaDevices.getUserMedia !== 'undefined';
    }

    /**
     * Check current microphone permission state
     */
    public static async checkPermission(): Promise<PermissionState> {
        if (typeof navigator === 'undefined' || !navigator.permissions) {
            return 'prompt';
        }
        try {
            // Try the standard permissions API
            const result = await navigator.permissions.query({ name: 'microphone' as PermissionName });
            return result.state;
        } catch (err) {
            // Fallback for browsers that don't support permissions API (e.g., Safari)
            // We'll detect the actual state when getUserMedia is called
            return 'prompt';
        }
    }

    /**
     * Get user-friendly error message from error
     */
    private getErrorMessage(err: any): string {
        if (!err) return "Unknown error occurred";
        
        const errorName = err.name || '';
        const errorMessage = err.message || '';

        // Permission denied errors
        if (errorName === 'NotAllowedError' || errorName === 'PermissionDeniedError' || 
            errorMessage.includes('permission') || errorMessage.includes('denied')) {
            return "Microphone permission denied. Please allow microphone access in your browser settings and try again.";
        }

        // No microphone found
        if (errorName === 'NotFoundError' || errorName === 'DevicesNotFoundError' ||
            errorMessage.includes('device') || errorMessage.includes('not found')) {
            return "No microphone found. Please connect a microphone and try again.";
        }

        // Browser not supported
        if (errorName === 'NotSupportedError' || errorMessage.includes('not supported')) {
            return "Your browser doesn't support audio recording. Please use Chrome, Firefox, Edge, or Safari (14.1+).";
        }

        // Constraint errors
        if (errorName === 'OverconstrainedError' || errorName === 'ConstraintNotSatisfiedError') {
            return "Microphone settings not supported. Try using a different microphone or browser.";
        }

        // Generic error
        return `Recording failed: ${errorMessage || errorName || 'Unknown error'}`;
    }

    /**
     * Verify that the audio stream is active
     */
    private verifyStreamActive(): boolean {
        if (!this.stream) return false;
        const tracks = this.stream.getAudioTracks();
        if (tracks.length === 0) return false;
        
        const activeTrack = tracks.find(track => track.readyState === 'live' && track.enabled);
        return !!activeTrack;
    }

    /**
     * Check for audio clipping using analyser node
     */
    private checkClipping(): void {
        if (!this.analyser) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);

        // Check for clipping (values near 0 or 255 indicate clipping in 8-bit)
        let clippingDetected = false;
        for (let i = 0; i < bufferLength; i++) {
            const value = dataArray[i];
            // Clipping occurs when values are at the extremes (0 or 255)
            // We check for values very close to these extremes
            if (value <= 2 || value >= 253) {
                clippingDetected = true;
                break;
            }
        }

        this.events.onClippingDetected?.(clippingDetected);
    }

    /**
     * Monitor stream activity, duration, and clipping
     */
    private startMonitoring(): void {
        this.startTime = Date.now();

        // Monitor stream activity
        this.streamCheckInterval = setInterval(() => {
            const isActive = this.verifyStreamActive();
            this.events.onStreamActive?.(isActive);
            
            if (!isActive && this.mediaRecorder?.state === 'recording') {
                console.warn('Audio stream became inactive during recording');
                this.events.onError?.(new Error('Audio stream stopped unexpectedly. Check your microphone connection.'));
            }
        }, 1000);

        // Update duration
        this.durationInterval = setInterval(() => {
            const duration = (Date.now() - this.startTime) / 1000;
            this.events.onDurationUpdate?.(duration);
        }, 100);

        // Check for clipping
        this.clippingCheckInterval = setInterval(() => {
            this.checkClipping();
        }, 200); // Check every 200ms
    }

    private stopMonitoring(): void {
        if (this.streamCheckInterval) {
            clearInterval(this.streamCheckInterval);
            this.streamCheckInterval = null;
        }
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
            this.durationInterval = null;
        }
        if (this.clippingCheckInterval) {
            clearInterval(this.clippingCheckInterval);
            this.clippingCheckInterval = null;
        }
    }

    public async startRecording(): Promise<void> {
        // Check API support
        if (!AudioRecorder.isSupported()) {
            const error = new Error("MediaRecorder API not supported in this browser");
            this.events.onError?.(error);
            throw error;
        }

        try {
            // Request microphone access with better error handling
            const constraints: MediaStreamConstraints = {
                audio: { 
                    echoCancellation: true, 
                    noiseSuppression: true, 
                    channelCount: 1,
                    sampleRate: { ideal: 44100, min: 16000 } // Request good quality
                },
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            // Verify stream is active
            if (!this.verifyStreamActive()) {
                throw new Error("Microphone stream is not active");
            }

            // Get actual audio settings for logging
            const audioTrack = this.stream.getAudioTracks()[0];
            const settings = audioTrack.getSettings();
            console.log('Audio recording settings:', {
                sampleRate: settings.sampleRate,
                channelCount: settings.channelCount,
                deviceId: settings.deviceId,
            });

            // Init AudioContext and analyser
            this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            this.sourceNode = this.audioContext.createMediaStreamSource(this.stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.sourceNode.connect(this.analyser);

            // MediaRecorder to capture raw PCM in browser-chosen mime
            const options: MediaRecorderOptions = {};
            // Prefer webm/ogg with Opus when possible; fallback to default
            if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
                options.mimeType = "audio/webm;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/webm")) {
                options.mimeType = "audio/webm";
            } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
                options.mimeType = "audio/ogg;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/ogg")) {
                options.mimeType = "audio/ogg";
            }

            this.mediaRecorder = new MediaRecorder(this.stream, options);

            this.chunks = [];
            
            // Enhanced data collection to prevent data loss
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    this.chunks.push(e.data);
                    this.events.onChunk?.(e.data);
                }
            };

            this.mediaRecorder.onstart = () => {
                this.startMonitoring();
                this.events.onStart?.();
            };

            this.mediaRecorder.onerror = (ev) => {
                const error = new Error((ev as any).error?.message || "MediaRecorder error");
                this.events.onError?.(error);
            };

            this.mediaRecorder.onstop = () => {
                this.stopMonitoring();
                
                // Ensure we have data
                if (this.chunks.length === 0) {
                    this.events.onError?.(new Error("No audio data recorded. Please try again."));
                    return;
                }

                // Create blob from collected chunks
                const blob = new Blob(this.chunks, { type: this.chunks[0]?.type || "audio/webm" });
                this.events.onStop?.(blob);
            };

            // Start recording with timeslice for better data collection
            this.mediaRecorder.start(100); // Request data every 100ms
        } catch (err: any) {
            this.stopMonitoring();
            const friendlyError = new Error(this.getErrorMessage(err));
            this.events.onError?.(friendlyError);
            throw friendlyError;
        }
    }

    public stopRecording(): void {
        try {
            this.stopMonitoring();
            
            // Stop MediaRecorder properly
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
                // onstop handler will be called automatically and will create the blob
            }
            
            // Stop all tracks
            if (this.stream) {
                this.stream.getTracks().forEach((t) => {
                    t.stop();
                    t.enabled = false;
                });
                this.stream = null;
            }
            
            // Clean up audio nodes
            if (this.sourceNode) {
                try {
                    this.sourceNode.disconnect();
                } catch (e) {
                    // Already disconnected
                }
                this.sourceNode = null;
            }
            if (this.analyser) {
                try {
                    this.analyser.disconnect();
                } catch (e) {
                    // Already disconnected
                }
                this.analyser = null;
            }
            if (this.audioContext) {
                this.audioContext.close().catch(console.error);
                this.audioContext = null;
            }
        } catch (err: any) {
            this.events.onError?.(err);
        }
    }

    /** Return the last recorded final Blob (webm/ogg) by concatenating chunks */
    public getLastBlob(): Blob | null {
        if (!this.chunks || this.chunks.length === 0) return null;
        return new Blob(this.chunks, { type: this.chunks[0].type });
    }
}
