
export type RecorderEvents = {
    onChunk?: (blobPart: Blob) => void;
    onStart?: () => void;
    onStop?: (finalBlob: Blob) => void;
    onError?: (err: Error) => void;
};

export class AudioRecorder {
    private mediaRecorder: MediaRecorder | null = null;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private stream: MediaStream | null = null;
    private chunks: Blob[] = [];
    private sourceNode: MediaStreamAudioSourceNode | null = null;

    constructor(private events: RecorderEvents = {}) {}

    public getAnalyser(): AnalyserNode | null {
        return this.analyser;
    }

    public async startRecording(): Promise<void> {
        try {
            // Request microphone access. Note: constraints hint desired sampleRate but actual may vary.
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
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
            if (MediaRecorder.isTypeSupported("audio/webm")) options.mimeType = "audio/webm";
            else if (MediaRecorder.isTypeSupported("audio/ogg")) options.mimeType = "audio/ogg";
            this.mediaRecorder = new MediaRecorder(this.stream!, options);

            this.chunks = [];
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size) {
                    this.chunks.push(e.data);
                    this.events.onChunk?.(e.data);
                }
            };

            this.mediaRecorder.onstart = () => this.events.onStart?.();
            this.mediaRecorder.onerror = (ev) =>
                this.events.onError?.(new Error((ev as any).error?.message || "MediaRecorder error"));
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.chunks, { type: this.chunks[0]?.type || "audio/webm" });
                this.events.onStop?.(blob);
            };

            this.mediaRecorder.start();
        } catch (err: any) {
            this.events.onError?.(err);
            throw err;
        }
    }

    public stopRecording(): void {
        try {
            if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
                this.mediaRecorder.stop();
            }
            if (this.stream) {
                this.stream.getTracks().forEach((t) => t.stop());
                this.stream = null;
            }
            if (this.sourceNode) {
                this.sourceNode.disconnect();
                this.sourceNode = null;
            }
            if (this.analyser) {
                this.analyser.disconnect();
                this.analyser = null;
            }
            if (this.audioContext) {
                this.audioContext.close();
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
