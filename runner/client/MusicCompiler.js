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

        // Expand loop constructs before parsing
        dslCode = this._expandLoops(dslCode);

        console.log('[MusicCompiler] Expanded DSL length:', dslCode.length);

        // Parse tempo
        const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
        const tempo = tempoMatch ? parseInt(tempoMatch[1]) : 120;

        // Parse tracks - use a helper function to handle brace matching
        const trackMatches = this._extractTracks(dslCode);

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

            // Parse 4-parameter notes with absolute timing
            const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
            if (noteMatches) {
                noteMatches.forEach(noteMatch => {
                    const [, note, start, duration, velocity] =
                        noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);

                    // Convert drum names to MIDI notes if needed
                    const resolvedNote = this._drumNameToNote(note);

                    this.scheduler.scheduleNote(
                        instrument,
                        resolvedNote,
                        parseFloat(duration),
                        parseFloat(start),
                        parseFloat(velocity)
                    );
                });
            }

            // Parse 4-parameter chords with absolute timing
            const chordMatches = trackMatch.match(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
            if (chordMatches) {
                chordMatches.forEach(chordMatch => {
                    const [, notesStr, start, duration, velocity] =
                        chordMatch.match(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);

                    const notes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));

                    this.scheduler.scheduleChord(
                        instrument,
                        notes,
                        parseFloat(duration),
                        parseFloat(start),
                        parseFloat(velocity)
                    );
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

            // Handle audio clips (vocal/audio tracks)
            if (track.audio) {
                track.audio.forEach(audioClip => {
                    this.scheduler.scheduleAudioClip(
                        audioClip.audio_data,
                        audioClip.start,
                        audioClip.duration,
                        audioClip.volume || 1.0,
                        track.id
                    );
                });
            }
        });

        return { tempo, duration: this.scheduler.getTotalDuration() };
    }

    /**
     * Extract track blocks from DSL with proper brace matching
     */
    _extractTracks(dslCode) {
        const tracks = [];
        const trackPattern = /track\("([^"]+)"\)\s*\{/g;
        let match;

        while ((match = trackPattern.exec(dslCode)) !== null) {
            const trackStart = match.index;
            const contentStart = trackPattern.lastIndex;

            // Find the matching closing brace
            let braceCount = 1;
            let pos = contentStart;

            while (pos < dslCode.length && braceCount > 0) {
                if (dslCode[pos] === '{') braceCount++;
                if (dslCode[pos] === '}') braceCount--;
                pos++;
            }

            if (braceCount === 0) {
                const trackBlock = dslCode.substring(trackStart, pos);
                tracks.push(trackBlock);
            }
        }

        return tracks.length > 0 ? tracks : null;
    }

    /**
     * Expand loop constructs in DSL code
     * Supports two syntaxes:
     * 1. Simple repeat: loop(count) { note(...) }
     * 2. Time-based: loop(startTime, endTime) { note(pitch, relativeStart, duration, velocity) }
     */
    _expandLoops(dslCode) {
        let expandedCode = dslCode;
        let maxIterations = 100;
        let iteration = 0;

        // Pattern for both simple and time-based loops
        const loopPattern = /(?:loop|for|while)\s*\(\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)\s*\{([^}]*)\}/g;
        let match;

        while ((match = loopPattern.exec(expandedCode)) !== null && iteration < maxIterations) {
            const fullMatch = match[0];
            const param1 = parseFloat(match[1]);
            const param2 = match[2] ? parseFloat(match[2]) : null;
            const loopContent = match[3];

            let expandedContent = '';

            if (param2 === null) {
                // Simple repeat: loop(N) - repeat N times
                for (let i = 0; i < param1; i++) {
                    expandedContent += loopContent;
                }
            } else {
                // Time-based: loop(startTime, endTime)
                const startTime = param1;
                const endTime = param2;

                // Parse notes and chords inside to get pattern duration
                const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
                const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;

                let noteMatches = [...loopContent.matchAll(notePattern)];
                let chordMatches = [...loopContent.matchAll(chordPattern)];

                if (noteMatches.length === 0 && chordMatches.length === 0) {
                    // No notes or chords found, skip this loop
                    expandedCode = expandedCode.replace(fullMatch, loopContent);
                    loopPattern.lastIndex = 0;
                    iteration++;
                    continue;
                }

                // Find the pattern duration (max relativeStart + duration)
                let patternDuration = 0;

                noteMatches.forEach(noteMatch => {
                    const relativeStart = parseFloat(noteMatch[2]);
                    const duration = parseFloat(noteMatch[3]);
                    patternDuration = Math.max(patternDuration, relativeStart + duration);
                });

                chordMatches.forEach(chordMatch => {
                    const relativeStart = parseFloat(chordMatch[2]);
                    const duration = parseFloat(chordMatch[3]);
                    patternDuration = Math.max(patternDuration, relativeStart + duration);
                });

                // Generate repeated notes from startTime to endTime
                const loopDuration = endTime - startTime;
                const repetitions = Math.ceil(loopDuration / patternDuration);

                for (let rep = 0; rep < repetitions; rep++) {
                    const repStartTime = startTime + (rep * patternDuration);

                    // Only add notes that fit within the loop range
                    if (repStartTime >= endTime) break;

                    // Expand notes
                    noteMatches.forEach(noteMatch => {
                        const pitch = noteMatch[1];
                        const relativeStart = parseFloat(noteMatch[2]);
                        const duration = parseFloat(noteMatch[3]);
                        const velocity = parseFloat(noteMatch[4]);

                        const absoluteStart = repStartTime + relativeStart;

                        // Only include notes that start before endTime
                        if (absoluteStart < endTime) {
                            expandedContent += `  note("${pitch}", ${absoluteStart}, ${duration}, ${velocity})\n`;
                        }
                    });

                    // Expand chords
                    chordMatches.forEach(chordMatch => {
                        const notes = chordMatch[1];
                        const relativeStart = parseFloat(chordMatch[2]);
                        const duration = parseFloat(chordMatch[3]);
                        const velocity = parseFloat(chordMatch[4]);

                        const absoluteStart = repStartTime + relativeStart;

                        // Only include chords that start before endTime
                        if (absoluteStart < endTime) {
                            expandedContent += `  chord([${notes}], ${absoluteStart}, ${duration}, ${velocity})\n`;
                        }
                    });
                }
            }

            // Replace the loop with expanded content
            expandedCode = expandedCode.replace(fullMatch, expandedContent);
            loopPattern.lastIndex = 0;
            iteration++;
        }

        return expandedCode;
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
     * Convert drum name to MIDI note, or return as-is if already a MIDI note
     * Allows using "kick", "snare", "hihat" instead of "C2", "D2", "F#2" in DSL
     */
    _drumNameToNote(note) {
        // Complete drum mapping matching server.js
        const drumMap = {
            kick: 'C2',
            snare: 'D2',
            snare_rimshot: 'E2',
            snare_buzz: 'D#2',
            hihat_closed: 'F#2',
            hihat_open: 'A#2',
            hihat_pedal: 'G#2',
            tom: 'G2',
            crash: 'C#3',
            ride: 'D#3',
            // Aliases for convenience
            hihat: 'F#2',  // Default to closed
            'hihat-closed': 'F#2',
            'hihat-open': 'A#2',
            'hihat-pedal': 'G#2'
        };

        // If it's a drum name, convert it
        const lowerNote = note.toLowerCase();
        if (drumMap[lowerNote]) {
            return drumMap[lowerNote];
        }

        // Otherwise return as-is (already a MIDI note like "C2")
        return note;
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