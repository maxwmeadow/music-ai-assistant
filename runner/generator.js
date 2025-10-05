class DSLGenerator {
    generate(parsedData) {
        let dsl = `tempo(${parsedData.metadata.tempo})\n\n`;

        parsedData.tracks.forEach(track => {
            dsl += this.generateTrack(track);
        });

        return dsl;
    }

    generateTrack(track) {
        let trackCode = `track("${track.id}") {\n`;

        if (track.instrument) {
            trackCode += `  instrument("${track.instrument}")\n`;
        }

        if (track.notes) {
            track.notes.forEach(note => {
                const noteName = this.midiToNote(note.pitch);
                trackCode += `  note("${noteName}", ${note.duration}, ${note.velocity})\n`;
            });
        }

        if (track.samples) {
            track.samples.forEach(sample => {
                trackCode += `  ${sample.sample}(${sample.start})\n`;
            });
        }

        trackCode += `}\n\n`;
        return trackCode;
    }

    midiToNote(midi) {
        const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const octave = Math.floor(midi / 12) - 1;
        const note = notes[midi % 12];
        return note + octave;
    }

    compileDSLToToneJS(dslCode) {
        let toneCode = `
// Auto-generated Tone.js from DSL

// ===== InstrumentRegistry Class =====
class InstrumentRegistry {
    constructor() {
        this.instruments = new Map();
        this.initializeInstruments();
    }

    initializeInstruments() {
        // Synthesized instruments
        this.registerSynth('synth_lead', () => {
            return new Tone.MonoSynth({
                oscillator: { type: 'sawtooth' },
                envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 1 },
                filterEnvelope: { attack: 0.06, decay: 0.2, sustain: 0.5, release: 2, baseFrequency: 200, octaves: 7 }
            }).toDestination();
        });

        this.registerSynth('pad', () => {
            return new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: 'sine' },
                envelope: { attack: 0.5, decay: 0.3, sustain: 0.7, release: 2 }
            }).toDestination();
        });

        this.registerSynth('bass', () => {
            return new Tone.MonoSynth({
                oscillator: { type: 'square' },
                envelope: { attack: 0.01, decay: 0.2, sustain: 0.4, release: 0.3 },
                filter: { type: 'lowpass', frequency: 800 }
            }).toDestination();
        });

        this.registerSynth('bells', () => {
            return new Tone.MetalSynth({
                frequency: 200,
                envelope: { attack: 0.001, decay: 1.4, release: 0.2 },
                harmonicity: 5.1,
                modulationIndex: 32,
                resonance: 4000
            }).toDestination();
        });

        this.registerSynth('strings', () => {
            return new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: 'sawtooth' },
                envelope: { attack: 0.2, decay: 0.1, sustain: 0.8, release: 1.5 }
            }).toDestination();
        });

        // Register sampled instruments
        this.registerSampledInstrument('piano/grand_piano_k');
        this.registerSampledInstrument('piano/grand_piano_s_model_b_1895');
        this.registerSampledInstrument('piano/upright_piano_knight');
        this.registerSampledInstrument('piano/upright_piano_y');
        this.registerSampledInstrument('harpsichord/harpsichord_english');
        this.registerSampledInstrument('harpsichord/harpsichord_flemish');
        this.registerSampledInstrument('harpsichord/harpsichord_french');
        this.registerSampledInstrument('harpsichord/harpsichord_italian');
        this.registerSampledInstrument('harpsichord/harpsichord_unk');
        this.registerSampledInstrument('guitar/rjs_guitar_new_strings');
        this.registerSampledInstrument('guitar/rjs_guitar_old_strings');
        this.registerSampledInstrument('guitar/rjs_guitar_palm_muted_softly_strings');
        this.registerSampledInstrument('guitar/rjs_guitar_palm_muted_strings');
        this.registerSampledInstrument('bass/jp8000_sawbass');
        this.registerSampledInstrument('bass/jp8000_tribass');
        this.registerSampledInstrument('strings/nfo_chamber_strings_longs');
        this.registerSampledInstrument('brass/nfo_iso_brass_swells');
        this.registerSampledInstrument('strings/nfo_iso_celli_swells');
        this.registerSampledInstrument('strings/nfo_iso_viola_swells');
        this.registerSampledInstrument('strings/nfo_iso_violin_swells');
        this.registerSampledInstrument('winds/nfo_iso_wind_swells');
        this.registerSampledInstrument('drums/lorenzos_drums');
        this.registerSampledInstrument('drums/bedroom_drums');
        this.registerSampledInstrument('synth/bass/2010_house');
        this.registerSampledInstrument('synth/bass/another_analog_bass');
        this.registerSampledInstrument('synth/bass/corg_bass');
        this.registerSampledInstrument('synth/bass/deep_undertone');
        this.registerSampledInstrument('synth/bass/lead_bass_player');
        this.registerSampledInstrument('synth/bass/ms20_bass');
        this.registerSampledInstrument('synth/bass/outrun_bass');
        this.registerSampledInstrument('synth/bass/thick_bass');
        this.registerSampledInstrument('synth/keys/dx_epiano');
        this.registerSampledInstrument('synth/keys/interstellar_on_a_budget');
        this.registerSampledInstrument('synth/keys/nord_string');
        this.registerSampledInstrument('synth/keys/outrun_pluck');
        this.registerSampledInstrument('synth/keys/rhode_less_traveled');
        this.registerSampledInstrument('synth/keys/synthetic_organ');
        this.registerSampledInstrument('synth/keys/synthetic_strings');
        this.registerSampledInstrument('synth/keys/the_organ_trail');
        this.registerSampledInstrument('synth/lead/classic_saws');
        this.registerSampledInstrument('synth/lead/crystaline_80s');
        this.registerSampledInstrument('synth/lead/cs80-ish');
        this.registerSampledInstrument('synth/lead/fm_hard_lead');
        this.registerSampledInstrument('synth/lead/for_each_loop');
        this.registerSampledInstrument('synth/lead/forever_80s');
        this.registerSampledInstrument('synth/lead/french_house');
        this.registerSampledInstrument('synth/lead/imfamousoty');
        this.registerSampledInstrument('synth/lead/jp_patchlead');
        this.registerSampledInstrument('synth/lead/juno_was_is');
        this.registerSampledInstrument('synth/lead/legecy_lead');
        this.registerSampledInstrument('synth/lead/poptab');
        this.registerSampledInstrument('synth/lead/strangest_things');
        this.registerSampledInstrument('synth/lead/synthetic_brass');
        this.registerSampledInstrument('synth/lead/the_nord');
        this.registerSampledInstrument('synth/lead/the_stack_guitar_chug');
        this.registerSampledInstrument('synth/lead/the_stack_guitar');
        this.registerSampledInstrument('synth/lead/uberheim_legend');
        this.registerSampledInstrument('synth/pad/airlock_leak');
        this.registerSampledInstrument('synth/pad/event_horizon');
        this.registerSampledInstrument('synth/pad/every_80s_movie_ever');
        this.registerSampledInstrument('synth/pad/fatness_pad');
        this.registerSampledInstrument('synth/pad/on_the_horizon');
        this.registerSampledInstrument('synth/pad/orion_belt');
        this.registerSampledInstrument('synth/pad/soft_and_padded');
        this.registerSampledInstrument('synth/pad/the_first_pad');
        this.registerSampledInstrument('synth/pad/timeless_movement');
        this.registerSampledInstrument('synth/seq/rhythmic_seq_1_(100bpm)');
        this.registerSampledInstrument('synth/seq/rhythmic_seq_2_(100bpm)');
    }

    registerSynth(name, factory) {
        this.instruments.set(name, {
            type: 'synth',
            factory: factory
        });
    }

    registerSampledInstrument(path) {
        this.instruments.set(path, {
            type: 'sampled',
            path: path
        });
    }

    async createInstrument(name) {
        const config = this.instruments.get(name);
        if (!config) {
            throw new Error(\`Unknown instrument: \${name}\`);
        }
        
        if (config.type === 'synth') {
            return config.factory();
        } else if (config.type === 'sampled') {
            return await this.createSampledInstrument(config.path);
        }
        
        throw new Error(\`Instrument type not supported: \${config.type}\`);
    }

    async createSampledInstrument(instrumentPath) {
        console.log(\`Loading sampled instrument: \${instrumentPath}\`);
        
        try {
            // CDN URL for R2
            const CDN_BASE = 'https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev';
            
            // Load mapping.json from R2
            const mappingPath = \`\${CDN_BASE}/samples/\${instrumentPath}/mapping.json\`;
            const cacheBuster = '?v=\${Date.now()}';
            console.log(\`[DEBUG] Fetching mapping from: \${mappingPath}\${cacheBuster}\`);
            const mappingResponse = await fetch(mappingPath + cacheBuster, {
                cache: 'no-store'
            });
    
            if (!mappingResponse.ok) {
                throw new Error(\`Failed to load mapping: \${mappingPath}\`);}
    
            const mapping = await mappingResponse.json();
            console.log(\`[DEBUG] Mapping loaded successfully\`);
            console.log(\`[DEBUG] Velocity layers keys:\`, Object.keys(mapping.velocity_layers));
    
            // Build sample URLs from velocity layers
            const urls = {};
            const baseUrl = \`\${CDN_BASE}/samples/\${instrumentPath}/\`;
            console.log(\`[DEBUG] Base URL: \${baseUrl}\`);
    
            // Use velocity_layers to build the sampler
            for (const [note, layers] of Object.entries(mapping.velocity_layers)) {
                console.log(\`[DEBUG] Processing note: $/{note}, layers count: $/{layers.length}\`);
                
                let sampleKey = note; // Default: use the note name as-is
                
                // For drum instruments, map drum sound names to MIDI notes
                if (mapping.type === 'drums') {
                    const drumNoteMap = {
                        "kick": "C2",           // MIDI 36
                        "snare": "D2",          // MIDI 38
                        "snare_rimshot": "E2",  // MIDI 40
                        "snare_buzz": "D#2",    // MIDI 39
                        "hihat_closed": "F#2",  // MIDI 42
                        "tom": "G2",            // MIDI 43
                        "crash": "C#3",         // MIDI 49
                        "ride": "D#3",          // MIDI 51
                        "hihat_pedal": "G#2",   // MIDI 44
                        "hihat_open": "A#2"     // MIDI 46
                    };
                    
                    sampleKey = drumNoteMap[note] || note;
                    console.log(\`[DEBUG] Drum sound "/{note}" mapped to note "$/{sampleKey}"\`);
                }             
                
                // For pianos: prefer sustain samples
                const sustainLayer = layers.find(l => l.file.includes('Sustains') || l.file.includes('sus'));
    
                if (sustainLayer) {
                    console.log(\`[DEBUG]   Found sustain layer for \${note}:\`);
                    console.log(\`[DEBUG]     - file: \${sustainLayer.file}\`);
                    urls[sampleKey] = encodeURIComponent(sustainLayer.file).replace(/%2F/g, '/');
                    console.log(\`[DEBUG]     - Final URL will be: \${baseUrl}\${sustainLayer.file}\`);
                } else if (layers.length > 0) {
                    // For guitars and other instruments: use middle velocity (vel4)
                    const midVelLayer = layers.find(l => l.file.includes('vel4')) || layers[Math.floor(layers.length / 2)];
                    console.log(\`[DEBUG]   Using mid-velocity layer for \${note}:\`);
                    console.log(\`[DEBUG]     - file: \${midVelLayer.file}\`);
                    urls[sampleKey] = encodeURIComponent(midVelLayer.file).replace(/%2F/g, '/');
                    console.log(\`[DEBUG]     - Final URL will be: \${baseUrl}\${midVelLayer.file}\`);
                } else {
                    console.warn(\`[DEBUG]   No suitable layer found for \${note}\`);
                }
            }
    
            console.log(\`[DEBUG] Total notes mapped: \${Object.keys(urls).length}\`);
            console.log(\`[DEBUG] Sample of URLs object:\`, Object.entries(urls).slice(0, 3));
            console.log(\`Creating Tone.Sampler with \${Object.keys(urls).length} notes\`);
    
            // Return a promise that resolves when the sampler is loaded
            return new Promise((resolve, reject) => {
                const sampler = new Tone.Sampler({
                    urls: urls,
                    baseUrl: baseUrl,
                    volume: 18,
                    onload: () => {
                        console.log(\`Successfully loaded sampled instrument: \${instrumentPath}\`);
                        resolve(sampler);
                    },
                    onerror: (error) => {
                        console.error(\`Error loading sampled instrument: \${instrumentPath}\`, error);
                        reject(new Error(\`Failed to load samples for \${instrumentPath}\`));
                    }
                }).toDestination();
            });
        }
    
        catch(error) {
            console.error(\`Error creating sampled instrument: \${instrumentPath}\`, error);
            throw error;
        }
    }
}

// ===== InstrumentPool Class =====
class InstrumentPool {
    constructor(instrumentName, registry, poolSize = 8) {
        this.instrumentName = instrumentName;
        this.registry = registry;
        this.poolSize = poolSize;
        this.instruments = [];
        this.nextIndex = 0;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        console.log(\`Initializing pool for \${this.instrumentName} with \${this.poolSize} instances\`);
        
        for (let i = 0; i < this.poolSize; i++) {
            const instrument = await this.registry.createInstrument(this.instrumentName);
            this.instruments.push(instrument);
        }
        
        this.initialized = true;
    }

    getNextInstrument() {
        if (!this.initialized) {
            throw new Error('Pool not initialized. Call initialize() first.');
        }
        
        const instrument = this.instruments[this.nextIndex];
        this.nextIndex = (this.nextIndex + 1) % this.instruments.length;
        return instrument;
    }

    dispose() {
        this.instruments.forEach(inst => inst.dispose());
        this.instruments = [];
        this.initialized = false;
    }
}

// ===== Music Playback Function =====
const instrumentRegistry = new InstrumentRegistry();
const pools = new Map();

async function initializeInstruments(trackConfigs) {
    for (const [trackId, instrumentName] of Object.entries(trackConfigs)) {
        if (!pools.has(trackId)) {
            const pool = new InstrumentPool(instrumentName, instrumentRegistry, 8);
            await pool.initialize();
            pools.set(trackId, pool);
            console.log(\`Initialized pool for track \${trackId}\`);
        }
    }
}

async function playMusic() {
    console.log("Initializing music playback...");
    
    if (Tone.context.state !== 'running') {
        await Tone.start();
        console.log("Tone.js audio context started");
    }
    
    Tone.Transport.cancel();
    console.log("Cleared existing transport events");
    
    const trackConfigs = {};
`;

        const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
        if (tempoMatch) {
            toneCode += `    Tone.Transport.bpm.value = ${tempoMatch[1]};\n`;
        }

        const trackTimes = {};

        const trackMatches = dslCode.match(/track\("([^"]+)"\)\s*{([^}]+)}/g);
        if (trackMatches) {
            // First pass: collect instrument configs
            trackMatches.forEach(trackMatch => {
                const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
                const trackId = trackIdMatch[1];

                const instrumentMatch = trackMatch.match(/instrument\("([^"]+)"\)/);
                if (instrumentMatch) {
                    const instrumentName = instrumentMatch[1];
                    toneCode += `    trackConfigs["${trackId}"] = "${instrumentName}";\n`;
                }
            });

            toneCode += `\n    await initializeInstruments(trackConfigs);\n\n`;

            // Second pass: schedule notes with proper timing
            trackMatches.forEach(trackMatch => {
                const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
                const trackId = trackIdMatch[1];

                if (!trackTimes[trackId]) {
                    trackTimes[trackId] = 0;
                }

                const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/g);
                if (noteMatches) {
                    noteMatches.forEach((noteMatch) => {
                        const [, note, duration, velocity] = noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/);

                        toneCode += `    Tone.Transport.schedule((time) => {\n`;
                        toneCode += `        const instrument = pools.get("${trackId}").getNextInstrument();\n`;
                        toneCode += `        const instrumentName = trackConfigs["${trackId}"];\n`;
                        toneCode += `        \n`;
                        toneCode += `        let noteToPlay = "${note}";\n`;
                        toneCode += `        \n`;
                        toneCode += `        // Check if this is a drum instrument\n`;
                        toneCode += `        if (instrumentName && instrumentName.includes('drums/')) {\n`;
                        toneCode += `            const drumNoteMap = {\n`;
                        toneCode += `                "kick": "C2",\n`;
                        toneCode += `                "snare": "D2",\n`;
                        toneCode += `                "snare_rimshot": "E2",\n`;
                        toneCode += `                "snare_buzz": "D#2",\n`;
                        toneCode += `                "hihat_closed": "F#2",\n`;
                        toneCode += `                "tom": "G2",\n`;
                        toneCode += `                "crash": "C#3",\n`;
                        toneCode += `                "ride": "D#3",\n`;
                        toneCode += `                "hihat_pedal": "G#2",\n`;
                        toneCode += `                "hihat_open": "A#2"\n`;
                        toneCode += `            };\n`;
                        toneCode += `            noteToPlay = drumNoteMap["${note}"] || "C2";\n`;
                        toneCode += `            console.log("Drum ${note} mapped to " + noteToPlay);\n`;
                        toneCode += `        }\n`;
                        toneCode += `        \n`;
                        toneCode += `        instrument.triggerAttackRelease(noteToPlay, ${duration}, time, ${velocity});\n`;
                        toneCode += `    }, "${trackTimes[trackId]}");\n`;

                        trackTimes[trackId] += parseFloat(duration);
                    });
                }

                // Handle drum samples (existing code - keep this)
                const kickMatches = trackMatch.match(/kick\(([\d.]+)\)/g);
                if (kickMatches) {
                    kickMatches.forEach(kickMatch => {
                        const [, startTime] = kickMatch.match(/kick\(([\d.]+)\)/);
                        toneCode += `    Tone.Transport.schedule((time) => {\n`;
                        toneCode += `        const instrument = pools.get("${trackId}").getNextInstrument();\n`;
                        toneCode += `        if (instrument.kick) instrument.kick.triggerAttackRelease("C2", "8n", time);\n`;
                        toneCode += `        console.log("Playing kick at ${startTime}");\n`;
                        toneCode += `    }, "${startTime}");\n`;
                    });
                }

                const snareMatches = trackMatch.match(/snare\(([\d.]+)\)/g);
                if (snareMatches) {
                    snareMatches.forEach(snareMatch => {
                        const [, startTime] = snareMatch.match(/snare\(([\d.]+)\)/);
                        toneCode += `    Tone.Transport.schedule((time) => {\n`;
                        toneCode += `        const instrument = pools.get("${trackId}").getNextInstrument();\n`;
                        toneCode += `        if (instrument.snare) instrument.snare.triggerAttackRelease("4n", time);\n`;
                        toneCode += `        console.log("Playing snare at ${startTime}");\n`;
                        toneCode += `    }, "${startTime}");\n`;
                    });
                }
            });
        }

        const maxDuration = Math.max(...Object.values(trackTimes), 0);
        const playbackDuration = Math.ceil((maxDuration + 2) * 1000);

        toneCode += `
    Tone.Transport.start();
    console.log("Music playback started");
    
    setTimeout(() => {
        Tone.Transport.stop();
        console.log("Music playback finished");
    }, ${playbackDuration});
}

// Export function using Offline rendering for WAV compatibility
async function exportAudio() {
    console.log("Starting audio export...");
    
    // Use Tone.Offline to render to a buffer
    const buffer = await Tone.Offline(async ({ transport }) => {
        const trackConfigs = {};
        transport.bpm.value = ${tempoMatch ? tempoMatch[1] : 120};
`;

        // Re-add track configs for offline rendering
        if (trackMatches) {
            trackMatches.forEach(trackMatch => {
                const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
                const trackId = trackIdMatch[1];
                const instrumentMatch = trackMatch.match(/instrument\("([^"]+)"\)/);
                if (instrumentMatch) {
                    const instrumentName = instrumentMatch[1];
                    toneCode += `        trackConfigs["${trackId}"] = "${instrumentName}";\n`;
                }
            });
        }

        toneCode += `
        await initializeInstruments(trackConfigs);
`;

        // Re-schedule notes for offline rendering
        const exportTrackTimes = {};
        if (trackMatches) {
            trackMatches.forEach(trackMatch => {
                const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
                const trackId = trackIdMatch[1];

                if (!exportTrackTimes[trackId]) {
                    exportTrackTimes[trackId] = 0;
                }

                const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/g);
                if (noteMatches) {
                    noteMatches.forEach((noteMatch) => {
                        const [, note, duration, velocity] = noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/);

                        toneCode += `        transport.schedule((time) => {\n`;
                        toneCode += `            const instrument = pools.get("${trackId}").getNextInstrument();\n`;
                        toneCode += `            instrument.triggerAttackRelease("${note}", ${duration}, time, ${velocity});\n`;
                        toneCode += `        }, "${exportTrackTimes[trackId]}");\n`;

                        exportTrackTimes[trackId] += parseFloat(duration);
                    });
                }
            });
        }

        toneCode += `
        transport.start();
    }, ${maxDuration + 2});
    
    console.log("Rendering complete, converting to WAV...");
    
    // Convert to WAV
    const wav = await audioBufferToWav(buffer);
    const blob = new Blob([wav], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    
    const anchor = document.createElement("a");
    anchor.download = "music-" + Date.now() + ".wav";
    anchor.href = url;
    anchor.click();
    
    console.log("Export complete");
}

// Helper function to convert AudioBuffer to WAV
function audioBufferToWav(buffer) {
    const length = buffer.length * buffer.numberOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let offset = 0;
    let pos = 0;
    
    // Write WAV header
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"
    
    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16); // length = 16
    setUint16(1); // PCM (uncompressed)
    setUint16(buffer.numberOfChannels);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * buffer.numberOfChannels); // avg. bytes/sec
    setUint16(buffer.numberOfChannels * 2); // block-align
    setUint16(16); // 16-bit
    
    setUint32(0x61746164); // "data" - chunk
    setUint32(length - pos - 4); // chunk length
    
    // Write audio data
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
    
    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }
    
    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }
}

window.exportAudio = exportAudio;

playMusic().catch(console.error);
`;

        return toneCode;
    }
}

module.exports = DSLGenerator;