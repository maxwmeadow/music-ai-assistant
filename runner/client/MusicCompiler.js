/**
 * MusicCompiler - Core runner compilation system
 * Takes DSL from frontend/backend and compiles to playable Tone.js
 * This replaces the old generator.js approach
 */

import sampleCache from './SampleCache.js';
import instrumentFactory from './InstrumentFactory.js';
import MusicScheduler from './MusicScheduler.js';

class MusicCompiler {
    constructor() {
        this.scheduler = new MusicScheduler();
        this.trackInstruments = new Map(); // trackId -> instrument
        this.initialized = false;
    }

    /**
     * Initialize the compiler with catalog
     */
    async initialize() {
        if (this.initialized) return;

        await sampleCache.loadCatalog();
        this.initialized = true;
        console.log('[MusicCompiler] Initialized');
    }

    /**
     * Compile from DSL code
     */
    async compileDSL(dslCode) {
        await this.initialize();

        console.log('[MusicCompiler] Compiling DSL...');

        // Parse tempo
        const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
        const tempo = tempoMatch ? parseInt(tempoMatch[1]) : 120;

        // Parse tracks
        const trackMatches = dslCode.match(/track\("([^"]+)"\)\s*{([^}]+)}/g);

        if (!trackMatches) {
            throw new Error('No tracks found in DSL');
        }

        // First pass: create all instruments
        const trackConfigs = [];

        for (const trackMatch of trackMatches) {
            const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
            const trackId = trackIdMatch[1];

            const instrumentMatch = trackMatch.match(/instrument\("([^"]+)"\)/);
            if (instrumentMatch) {
                const instrumentName = instrumentMatch[1];
                trackConfigs.push({ trackId, instrumentName });
            }
        }

        // Create instruments with polyphonic pools
        await this._createInstruments(trackConfigs);

        // Second pass: schedule notes
        let trackTimes = {};

        for (const trackMatch of trackMatches) {
            const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
            const trackId = trackIdMatch[1];
            const instrument = this.trackInstruments.get(trackId);

            if (!instrument) {
                console.warn(`[MusicCompiler] No instrument for track ${trackId}`);
                continue;
            }

            if (!trackTimes[trackId]) {
                trackTimes[trackId] = 0;
            }

            // Parse notes
            const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/g);
            if (noteMatches) {
                noteMatches.forEach(noteMatch => {
                    const [, note, duration, velocity] =
                        noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/);

                    this.scheduler.scheduleNote(
                        instrument,
                        note,
                        parseFloat(duration),
                        trackTimes[trackId],
                        parseFloat(velocity)
                    );

                    trackTimes[trackId] += parseFloat(duration);
                });
            }

            // Parse chord notation (if exists)
            const chordMatches = trackMatch.match(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+)\)/g);
            if (chordMatches) {
                chordMatches.forEach(chordMatch => {
                    const [, notesStr, duration, velocity] =
                        chordMatch.match(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+)\)/);

                    const notes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));

                    this.scheduler.scheduleChord(
                        instrument,
                        notes,
                        parseFloat(duration),
                        trackTimes[trackId],
                        parseFloat(velocity)
                    );

                    trackTimes[trackId] += parseFloat(duration);
                });
            }
        }

        return { tempo, duration: this.scheduler.getTotalDuration() };
    }

    /**
     * Compile from IR (Intermediate Representation)
     */
    async compileIR(irData) {
        await this.initialize();

        console.log('[MusicCompiler] Compiling IR...');

        const tempo = irData.metadata?.tempo || 120;

        // Create instruments for all tracks
        const trackConfigs = irData.tracks.map(track => ({
            trackId: track.id,
            instrumentName: track.instrument
        }));

        await this._createInstruments(trackConfigs);

        // Schedule notes from each track
        irData.tracks.forEach(track => {
            const instrument = this.trackInstruments.get(track.id);
            if (!instrument) return;

            let currentTime = 0;

            // Handle notes
            if (track.notes) {
                track.notes.forEach(noteData => {
                    const note = this._midiToNote(noteData.pitch);

                    // Check if this is a chord (array of pitches)
                    if (Array.isArray(noteData.pitch)) {
                        const notes = noteData.pitch.map(p => this._midiToNote(p));
                        this.scheduler.scheduleChord(
                            instrument,
                            notes,
                            noteData.duration,
                            currentTime,
                            noteData.velocity || 1
                        );
                    } else {
                        this.scheduler.scheduleNote(
                            instrument,
                            note,
                            noteData.duration,
                            currentTime,
                            noteData.velocity || 1
                        );
                    }

                    currentTime += noteData.duration;
                });
            }

            // Handle samples (drums, etc.)
            if (track.samples) {
                track.samples.forEach(sample => {
                    // Map sample type to note
                    const note = this._sampleToNote(sample.sample);
                    this.scheduler.scheduleNote(
                        instrument,
                        note,
                        0.5, // Default drum hit duration
                        sample.start,
                        sample.velocity || 1
                    );
                });
            }
        });

        return { tempo, duration: this.scheduler.getTotalDuration() };
    }

    /**
     * Create instruments for all tracks
     */
    async _createInstruments(trackConfigs) {
        console.log(`[MusicCompiler] Creating ${trackConfigs.length} instruments...`);

        const loadPromises = trackConfigs.map(async ({ trackId, instrumentName }) => {
            try {
                // Create polyphonic pool for better chord support
                const instrument = await instrumentFactory.createPolyphonicPool(
                    instrumentName,
                    8, // 8 voices
                    { preload: false } // Don't preload by default
                );

                this.trackInstruments.set(trackId, instrument);
                console.log(`[MusicCompiler] Created instrument for track ${trackId}: ${instrumentName}`);
            } catch (error) {
                console.error(`[MusicCompiler] Failed to create instrument for ${trackId}:`, error);
            }
        });

        await Promise.all(loadPromises);
    }

    /**
     * Convert MIDI note number to note name
     */
    _midiToNote(midi) {
        const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const octave = Math.floor(midi / 12) - 1;
        const note = notes[midi % 12];
        return note + octave;
    }

    /**
     * Map sample names to notes (for drums)
     */
    _sampleToNote(sampleName) {
        const drumMap = {
            kick: 'C2',
            snare: 'D2',
            hihat: 'F#2',
            hihat_closed: 'F#2',
            hihat_open: 'A#2',
            tom: 'G2',
            crash: 'C#3',
            ride: 'D#3'
        };
        return drumMap[sampleName] || 'C2';
    }

    /**
     * Start playback
     */
    async play() {
        const stats = await this.compileIR(irData);
        await this.scheduler.start(stats.tempo);

        // Auto-stop after duration
        setTimeout(() => {
            this.stop();
        }, (stats.duration + 1) * 1000);
    }

    /**
     * Stop playback
     */
    stop() {
        this.scheduler.stop();
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.scheduler.clear();
        instrumentFactory.disposeAll();
        this.trackInstruments.clear();
        console.log('[MusicCompiler] Disposed all resources');
    }

    /**
     * Export to WAV file
     */
    async exportWAV() {
        console.log('[MusicCompiler] Exporting to WAV...');

        const buffer = await this.scheduler.renderToBuffer();
        const wav = this._audioBufferToWav(buffer);
        const blob = new Blob([wav], { type: 'audio/wav' });

        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.download = `music-${Date.now()}.wav`;
        anchor.href = url;
        anchor.click();

        console.log('[MusicCompiler] Export complete');
    }

    /**
     * Convert AudioBuffer to WAV format
     */
    _audioBufferToWav(buffer) {
        const length = buffer.length * buffer.numberOfChannels * 2 + 44;
        const arrayBuffer = new ArrayBuffer(length);
        const view = new DataView(arrayBuffer);
        const channels = [];
        let offset = 0;
        let pos = 0;

        // WAV header
        const setUint16 = (data) => { view.setUint16(pos, data, true); pos += 2; };
        const setUint32 = (data) => { view.setUint32(pos, data, true); pos += 4; };

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8);
        setUint32(0x45564157); // "WAVE"
        setUint32(0x20746d66); // "fmt "
        setUint32(16);
        setUint16(1); // PCM
        setUint16(buffer.numberOfChannels);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * buffer.numberOfChannels);
        setUint16(buffer.numberOfChannels * 2);
        setUint16(16);
        setUint32(0x61746164); // "data"
        setUint32(length - pos - 4);

        // Audio data
        for (let i = 0; i < buffer.numberOfChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }

        while (pos < length) {
            for (let i = 0; i < buffer.numberOfChannels; i++) {
                let sample = Math.max(-1, Math.min(1, channels[i][offset]));
                sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                view.setInt16(pos, sample, true);
                pos += 2;
            }
            offset++;
        }

        return arrayBuffer;
    }

    /**
     * Get compiler statistics
     */
    getStats() {
        return {
            initialized: this.initialized,
            trackCount: this.trackInstruments.size,
            scheduler: this.scheduler.getStats(),
            factory: instrumentFactory.getStats()
        };
    }
}

export default MusicCompiler;