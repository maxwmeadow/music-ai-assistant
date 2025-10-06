/**
 * InstrumentFactory.js - Smart Instrument Creation
 * Handles both synthesized and sampled instruments with polyphonic pools
 */

import sampleCache from './SampleCache.js';

class InstrumentFactory {
    constructor() {
        this.synthFactories = new Map();
        this.instrumentCache = new Map();
        this.initializeSynthFactories();
    }

    initializeSynthFactories() {
        this.synthFactories.set('synth_lead', () =>
            new Tone.MonoSynth({
                oscillator: { type: 'sawtooth' },
                envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 1 },
                filterEnvelope: {
                    attack: 0.06, decay: 0.2, sustain: 0.5, release: 2,
                    baseFrequency: 200, octaves: 7
                }
            }).toDestination()
        );

        this.synthFactories.set('pad', () =>
            new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: 'sine' },
                envelope: { attack: 0.5, decay: 0.3, sustain: 0.7, release: 2 }
            }).toDestination()
        );

        this.synthFactories.set('bass', () =>
            new Tone.MonoSynth({
                oscillator: { type: 'square' },
                envelope: { attack: 0.01, decay: 0.2, sustain: 0.4, release: 0.3 },
                filter: { type: 'lowpass', frequency: 800 }
            }).toDestination()
        );

        this.synthFactories.set('bells', () =>
            new Tone.MetalSynth({
                frequency: 200,
                envelope: { attack: 0.001, decay: 1.4, release: 0.2 },
                harmonicity: 5.1,
                modulationIndex: 32,
                resonance: 4000
            }).toDestination()
        );

        this.synthFactories.set('strings', () =>
            new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: 'sawtooth' },
                envelope: { attack: 0.2, decay: 0.1, sustain: 0.8, release: 1.5 }
            }).toDestination()
        );
    }

    async createInstrument(instrumentPath, options = {}) {
        const cacheKey = `${instrumentPath}-${JSON.stringify(options)}`;

        if (this.instrumentCache.has(cacheKey) && !options.forceNew) {
            console.log(`[InstrumentFactory] Using cached: ${instrumentPath}`);
            return this.instrumentCache.get(cacheKey);
        }

        if (this.synthFactories.has(instrumentPath)) {
            const instrument = this.synthFactories.get(instrumentPath)();
            if (!options.forceNew) {
                this.instrumentCache.set(cacheKey, instrument);
            }
            return instrument;
        }

        const instrument = await this.createSampledInstrument(instrumentPath, options);
        if (!options.forceNew) {
            this.instrumentCache.set(cacheKey, instrument);
        }
        return instrument;
    }

    async createSampledInstrument(instrumentPath, options = {}) {
        console.log(`[InstrumentFactory] Creating sampled instrument: ${instrumentPath}`);

        try {
            const mapping = await sampleCache.getMapping(instrumentPath);
            const { urls, baseUrl } = sampleCache.buildSamplerUrls(mapping, instrumentPath);

            if (options.preload) {
                await sampleCache.preloadSamples(urls, baseUrl);
            }

            return new Promise((resolve, reject) => {
                const sampler = new Tone.Sampler({
                    urls,
                    baseUrl,
                    volume: options.volume ?? 0,
                    attack: options.attack ?? 0,
                    release: options.release ?? 0.5,
                    onload: () => {
                        console.log(`[InstrumentFactory] Loaded: ${instrumentPath}`);
                        resolve(sampler);
                    },
                    onerror: (error) => {
                        console.error(`[InstrumentFactory] Error loading ${instrumentPath}:`, error);
                        reject(error);
                    }
                }).toDestination();
            });
        } catch (error) {
            console.error(`[InstrumentFactory] Failed to create ${instrumentPath}:`, error);
            throw error;
        }
    }

    async createPolyphonicPool(instrumentPath, voiceCount = 8, options = {}) {
        console.log(`[InstrumentFactory] Creating polyphonic pool: ${instrumentPath} (${voiceCount} voices)`);

        const pool = [];
        const baseInstrument = await this.createInstrument(instrumentPath, options);
        pool.push(baseInstrument);

        for (let i = 1; i < voiceCount; i++) {
            const voice = await this.createInstrument(instrumentPath, {
                ...options,
                forceNew: true
            });
            pool.push(voice);
        }

        return new PolyphonicInstrumentPool(pool, instrumentPath);
    }

    disposeInstrument(instrumentPath) {
        if (this.instrumentCache.has(instrumentPath)) {
            const instrument = this.instrumentCache.get(instrumentPath);
            instrument.dispose();
            this.instrumentCache.delete(instrumentPath);
            console.log(`[InstrumentFactory] Disposed: ${instrumentPath}`);
        }
    }

    disposeAll() {
        for (const [path, instrument] of this.instrumentCache) {
            instrument.dispose();
        }
        this.instrumentCache.clear();
        console.log('[InstrumentFactory] Disposed all instruments');
    }

    getStats() {
        return {
            cachedInstruments: this.instrumentCache.size,
            synthTypes: this.synthFactories.size,
            sampleCacheStats: sampleCache.getStats()
        };
    }
}

class PolyphonicInstrumentPool {
    constructor(instruments, name) {
        this.instruments = instruments;
        this.name = name;
        this.currentVoice = 0;
        this.activeNotes = new Map();
    }

    triggerAttackRelease(note, duration, time = Tone.now(), velocity = 1) {
        const voice = this.instruments[this.currentVoice];
        this.currentVoice = (this.currentVoice + 1) % this.instruments.length;

        const noteKey = `${note}-${time}`;
        this.activeNotes.set(noteKey, { voice: this.currentVoice, time });

        voice.triggerAttackRelease(note, duration, time, velocity);

        const durationSeconds = Tone.Time(duration).toSeconds();
        setTimeout(() => {
            this.activeNotes.delete(noteKey);
        }, (durationSeconds + 0.1) * 1000);
    }

    triggerChord(notes, duration, time = Tone.now(), velocity = 1) {
        console.log(`[PolyphonicPool] Playing chord:`, notes);
        notes.forEach(note => {
            this.triggerAttackRelease(note, duration, time, velocity);
        });
    }

    releaseAll(time = Tone.now()) {
        this.instruments.forEach(inst => {
            if (inst.triggerRelease) {
                inst.triggerRelease(time);
            }
        });
        this.activeNotes.clear();
    }

    dispose() {
        this.instruments.forEach(inst => inst.dispose());
        this.instruments = [];
        this.activeNotes.clear();
    }

    getStats() {
        return {
            name: this.name,
            voiceCount: this.instruments.length,
            activeNotes: this.activeNotes.size,
            currentVoice: this.currentVoice
        };
    }
}

const instrumentFactory = new InstrumentFactory();
export default instrumentFactory;
export { PolyphonicInstrumentPool };