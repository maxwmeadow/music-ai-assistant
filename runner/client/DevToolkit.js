/**
 * DevToolkit - Development and debugging utilities
 * Helps with testing, profiling, and troubleshooting
 */

class DevToolkit {
    constructor() {
        this.perfMarks = new Map();
        this.errorLog = [];
        this.debugMode = false;
    }

    /**
     * Enable debug mode with verbose logging
     */
    enableDebug() {
        this.debugMode = true;
        console.log('[DevToolkit] Debug mode enabled');
    }

    /**
     * Disable debug mode
     */
    disableDebug() {
        this.debugMode = false;
        console.log('[DevToolkit] Debug mode disabled');
    }

    /**
     * Performance timing helpers
     */
    startTiming(label) {
        this.perfMarks.set(label, performance.now());
        if (this.debugMode) {
            console.log(`[PERF] Started: ${label}`);
        }
    }

    endTiming(label) {
        const start = this.perfMarks.get(label);
        if (!start) {
            console.warn(`[PERF] No start mark for: ${label}`);
            return null;
        }

        const duration = performance.now() - start;
        this.perfMarks.delete(label);

        if (this.debugMode) {
            console.log(`[PERF] ${label}: ${duration.toFixed(2)}ms`);
        }

        return duration;
    }

    /**
     * Test DSL syntax
     */
    testDSL(dslCode) {
        const tests = {
            hasTempo: /tempo\(\d+\)/.test(dslCode),
            hasTracks: /track\("[^"]+"\)/.test(dslCode),
            hasInstruments: /instrument\("[^"]+"\)/.test(dslCode),
            hasNotes: /note\("[^"]+",\s*[\d.]+,\s*[\d.]+\)/.test(dslCode),
            balancedBraces: (dslCode.match(/{/g) || []).length === (dslCode.match(/}/g) || []).length
        };

        const issues = [];
        if (!tests.hasTempo) issues.push('Missing tempo declaration');
        if (!tests.hasTracks) issues.push('No tracks defined');
        if (!tests.hasInstruments) issues.push('No instruments specified');
        if (!tests.hasNotes) issues.push('No notes found');
        if (!tests.balancedBraces) issues.push('Unbalanced braces');

        return {
            valid: issues.length === 0,
            tests,
            issues
        };
    }

    /**
     * Test IR format
     */
    testIR(irData) {
        const tests = {
            hasMetadata: !!irData.metadata,
            hasTempo: !!irData.metadata?.tempo,
            hasTracks: Array.isArray(irData.tracks) && irData.tracks.length > 0,
            allTracksHaveIds: irData.tracks?.every(t => !!t.id) ?? false,
            allTracksHaveInstruments: irData.tracks?.every(t => !!t.instrument) ?? false,
            hasNotes: irData.tracks?.some(t => t.notes?.length > 0) ?? false
        };

        const issues = [];
        if (!tests.hasMetadata) issues.push('Missing metadata');
        if (!tests.hasTempo) issues.push('Missing tempo in metadata');
        if (!tests.hasTracks) issues.push('No tracks or invalid tracks array');
        if (!tests.allTracksHaveIds) issues.push('Some tracks missing IDs');
        if (!tests.allTracksHaveInstruments) issues.push('Some tracks missing instruments');
        if (!tests.hasNotes) issues.push('No notes in any track');

        return {
            valid: issues.length === 0,
            tests,
            issues,
            stats: {
                trackCount: irData.tracks?.length || 0,
                noteCount: irData.tracks?.reduce((sum, t) => sum + (t.notes?.length || 0), 0) || 0
            }
        };
    }

    /**
     * Analyze instrument loading performance
     */
    async profileInstrumentLoad(instrumentPath, sampleCache, instrumentFactory) {
        console.log(`[DevToolkit] Profiling instrument: ${instrumentPath}`);

        this.startTiming('total');

        // Test mapping load
        this.startTiming('mapping');
        const mapping = await sampleCache.getMapping(instrumentPath);
        const mappingTime = this.endTiming('mapping');

        // Test URL building
        this.startTiming('urlBuild');
        const { urls, baseUrl } = sampleCache.buildSamplerUrls(mapping, instrumentPath);
        const urlBuildTime = this.endTiming('urlBuild');

        // Test instrument creation
        this.startTiming('instrument');
        const instrument = await instrumentFactory.createInstrument(instrumentPath);
        const instrumentTime = this.endTiming('instrument');

        const totalTime = this.endTiming('total');

        const profile = {
            instrumentPath,
            timings: {
                mapping: `${mappingTime.toFixed(2)}ms`,
                urlBuild: `${urlBuildTime.toFixed(2)}ms`,
                instrument: `${instrumentTime.toFixed(2)}ms`,
                total: `${totalTime.toFixed(2)}ms`
            },
            sampleCount: Object.keys(urls).length,
            mapping: {
                type: mapping.type,
                noteCount: Object.keys(mapping.velocity_layers).length
            }
        };

        console.table(profile.timings);
        return profile;
    }

    /**
     * Compare multiple instruments
     */
    async compareInstruments(instrumentPaths, sampleCache, instrumentFactory) {
        console.log(`[DevToolkit] Comparing ${instrumentPaths.length} instruments...`);

        const results = [];

        for (const path of instrumentPaths) {
            const profile = await this.profileInstrumentLoad(path, sampleCache, instrumentFactory);
            results.push(profile);

            // Clean up
            instrumentFactory.disposeInstrument(path);
        }

        return results;
    }

    /**
     * Test chord playback
     */
    testChord(notes = ['C4', 'E4', 'G4']) {
        return {
            chord: notes,
            midiNumbers: notes.map(n => this._noteToMidi(n)),
            interval: this._analyzeIntervals(notes),
            chordName: this._identifyChord(notes)
        };
    }

    /**
     * Generate test patterns
     */
    generateTestPattern(type = 'scale') {
        const patterns = {
            scale: {
                name: 'C Major Scale',
                notes: ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
                duration: 0.5,
                velocity: 0.8
            },
            chord: {
                name: 'C Major Chord Progression',
                chords: [
                    ['C4', 'E4', 'G4'],
                    ['F4', 'A4', 'C5'],
                    ['G4', 'B4', 'D5'],
                    ['C4', 'E4', 'G4']
                ],
                duration: 1.0,
                velocity: 0.8
            },
            arpeggio: {
                name: 'C Major Arpeggio',
                notes: ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4'],
                duration: 0.25,
                velocity: 0.7
            },
            rhythm: {
                name: 'Basic Drum Pattern',
                samples: [
                    { sample: 'kick', start: 0 },
                    { sample: 'hihat', start: 0.25 },
                    { sample: 'snare', start: 0.5 },
                    { sample: 'hihat', start: 0.75 },
                    { sample: 'kick', start: 1.0 }
                ]
            }
        };

        return patterns[type] || patterns.scale;
    }

    /**
     * Monitor memory usage
     */
    getMemoryStats() {
        if (performance.memory) {
            return {
                usedJSHeapSize: `${(performance.memory.usedJSHeapSize / 1048576).toFixed(2)} MB`,
                totalJSHeapSize: `${(performance.memory.totalJSHeapSize / 1048576).toFixed(2)} MB`,
                jsHeapSizeLimit: `${(performance.memory.jsHeapSizeLimit / 1048576).toFixed(2)} MB`
            };
        }
        return { message: 'Memory API not available' };
    }

    /**
     * Log error with context
     */
    logError(error, context = {}) {
        const errorEntry = {
            timestamp: new Date().toISOString(),
            message: error.message,
            stack: error.stack,
            context
        };

        this.errorLog.push(errorEntry);
        console.error('[DevToolkit] Error logged:', errorEntry);

        return errorEntry;
    }

    /**
     * Get error report
     */
    getErrorReport() {
        return {
            errorCount: this.errorLog.length,
            errors: this.errorLog,
            memoryStats: this.getMemoryStats()
        };
    }

    /**
     * Clear error log
     */
    clearErrors() {
        this.errorLog = [];
        console.log('[DevToolkit] Error log cleared');
    }

    /**
     * Helper: Convert note name to MIDI number
     */
    _noteToMidi(note) {
        const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const match = note.match(/([A-G]#?)(\d+)/);
        if (!match) return null;

        const [, noteName, octave] = match;
        const noteIndex = notes.indexOf(noteName);
        return (parseInt(octave) + 1) * 12 + noteIndex;
    }

    /**
     * Helper: Analyze intervals in chord
     */
    _analyzeIntervals(notes) {
        const midiNotes = notes.map(n => this._noteToMidi(n)).sort((a, b) => a - b);
        const intervals = [];

        for (let i = 1; i < midiNotes.length; i++) {
            intervals.push(midiNotes[i] - midiNotes[i - 1]);
        }

        return intervals;
    }

    /**
     * Helper: Identify chord type
     */
    _identifyChord(notes) {
        const intervals = this._analyzeIntervals(notes);

        const chordTypes = {
            '4,3': 'Major',
            '3,4': 'Minor',
            '4,3,4': 'Major 7th',
            '3,4,3': 'Minor 7th',
            '4,3,3': 'Dominant 7th'
        };

        return chordTypes[intervals.join(',')] || 'Unknown';
    }

    /**
     * Export diagnostic report
     */
    exportDiagnostics(compiler) {
        const report = {
            timestamp: new Date().toISOString(),
            system: {
                userAgent: navigator.userAgent,
                memory: this.getMemoryStats(),
                audioContext: {
                    state: Tone.context.state,
                    sampleRate: Tone.context.sampleRate,
                    currentTime: Tone.context.currentTime
                }
            },
            compiler: compiler ? compiler.getStats() : null,
            errors: this.errorLog,
            performance: Object.fromEntries(this.perfMarks)
        };

        console.log('[DevToolkit] Diagnostic Report:', report);

        // Download as JSON
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `diagnostics-${Date.now()}.json`;
        a.click();

        return report;
    }
}

// Export singleton instance
const devToolkit = new DevToolkit();
export default devToolkit;