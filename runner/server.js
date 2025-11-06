/**
 * Runner Server - Music DSL Compilation Service
 * Receives DSL/IR from backend, compiles to executable Tone.js code
 * This replaces generator.js with the refactored system
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const MusicJSONParser = require('./parser');

// Note: For ES modules in Node.js, you'd use dynamic import
// For now, we'll structure this to work with your existing setup

const app = express();
const PORT = process.env.PORT || 5001;

// CORS configuration
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['http://localhost:3000', 'http://localhost:8000'];

console.log('Runner CORS allowed origins:', allowedOrigins);

// Middleware
app.use(cors({
  origin: allowedOrigins,
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));

// Static file serving for samples
app.use('/samples', express.static(path.join(__dirname, 'samples')));
app.use('/catalog.json', express.static(path.join(__dirname, 'catalog.json')));

// Serve the refactored client modules
app.use('/client', express.static(path.join(__dirname, 'client')));

// Request logging
app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.path}`);
    next();
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'music-runner',
        version: '2.0.0'
    });
});

/**
 * Main compilation endpoint
 * POST /eval
 * Body: { musicData: { ...IR or { __dsl_passthrough: "..." } } }
 * Returns: { dsl_code, executable_code, parsed_data }
 */
app.post('/eval', async (req, res) => {
    try {
        const { musicData } = req.body;

        if (!musicData) {
            return res.status(400).json({
                status: 'error',
                message: 'Missing musicData'
            });
        }

        console.log('[EVAL] Processing request...');

        let dslCode;
        let irData;

        // Check if DSL passthrough
        if (musicData.__dsl_passthrough) {
            console.log('[EVAL] DSL passthrough mode');
            dslCode = musicData.__dsl_passthrough;
            // Parse DSL to get metadata
            irData = null;
        } else {
            console.log('[EVAL] IR mode - converting to DSL');
            // Convert IR to DSL
            const parser = new MusicJSONParser();
            irData = parser.parse(JSON.stringify(musicData));
            dslCode = irToDSL(irData);
        }

        console.log('[EVAL] DSL Code:', dslCode.substring(0, 200) + '...');

        // Compile DSL to executable Tone.js code
        const result = await compileDSLToExecutable(dslCode);

        console.log('[EVAL] Compilation successful');

        res.json({
            status: 'success',
            dsl_code: result.dsl_code,
            executable_code: result.executable_code,
            parsed_data: result.parsed_data
        });

    } catch (error) {
        console.error('[EVAL] Error:', error);
        res.status(500).json({
            status: 'error',
            message: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

/**
 * Convert IR to DSL (matches compiler_stub.py logic)
 */
function irToDSL(ir) {
    const tempo = ir.metadata?.tempo || 120;
    let dsl = `tempo(${tempo})\n\n`;

    ir.tracks.forEach(track => {
        dsl += `track("${track.id}") {\n`;

        if (track.instrument) {
            dsl += `  instrument("${track.instrument}")\n`;
        }

        if (track.notes) {
            track.notes.forEach(note => {
                const noteName = midiToNote(note.pitch);
                dsl += `  note("${noteName}", ${note.start}, ${note.duration}, ${note.velocity})\n`;
            });
        }

        if (track.samples) {
            track.samples.forEach(sample => {
                dsl += `  ${sample.sample}(${sample.start})\n`;
            });
        }

        dsl += `}\n\n`;
    });

    return dsl;
}

/**
 * Compile DSL to executable Tone.js code
 * This is the core replacement for generator.js
 */
async function compileDSLToExecutable(dslCode) {
    const CDN_BASE = 'https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev';

    // Parse DSL
    const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
    const tempo = tempoMatch ? parseInt(tempoMatch[1]) : 120;

    const trackMatches = dslCode.match(/track\("([^"]+)"\)\s*{([^}]+)}/g);
    if (!trackMatches) {
        throw new Error('No tracks found in DSL');
    }

    const trackConfigs = [];
    const trackSchedules = [];

    // Parse each track
    for (const trackMatch of trackMatches) {
        const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
        const trackId = trackIdMatch[1];

        const instrumentMatch = trackMatch.match(/instrument\("([^"]+)"\)/);
        const instrumentName = instrumentMatch ? instrumentMatch[1] : null;

        if (instrumentName) {
            trackConfigs.push({ trackId, instrumentName });
        }

        // Parse notes
        const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
        const notes = [];

        if (noteMatches) {
            let currentTime = 0;
            noteMatches.forEach(noteMatch => {
                const [, note, start, duration, velocity] =
                    noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);

                notes.push({
                    note,
                    duration: parseFloat(duration),
                    velocity: parseFloat(velocity),
                    time: parseFloat(start)
                });

                currentTime += parseFloat(duration);
            });
        }

        // Parse chords
        const chordMatches = trackMatch.match(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
        const chords = [];

        if (chordMatches) {
            chordMatches.forEach(chordMatch => {
                const [, notesStr, start, duration, velocity] =
                    chordMatch.match(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);

                const chordNotes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));

                chords.push({
                    notes: chordNotes,
                    duration: parseFloat(duration),
                    velocity: parseFloat(velocity),
                    time: parseFloat(start)  // Use explicit start time
                });
            });
        }

        trackSchedules.push({
            trackId,
            instrumentName,
            notes,
            chords
        });
    }

    // Calculate max duration
    let maxDuration = 0;
    trackSchedules.forEach(schedule => {
        // Check notes
        schedule.notes.forEach(noteData => {
            const endTime = noteData.time + noteData.duration;
            maxDuration = Math.max(maxDuration, endTime);
        });

        // Check chords
        schedule.chords.forEach(chordData => {
            const endTime = chordData.time + chordData.duration;
            maxDuration = Math.max(maxDuration, endTime);
        });
    });

    maxDuration += 1;

    // Generate standalone executable code
    const executable_code = generateExecutableCode(
        CDN_BASE,
        tempo,
        trackConfigs,
        trackSchedules,
        maxDuration
    );

    return {
        dsl_code: dslCode,
        executable_code,
        parsed_data: {
            tempo,
            duration: maxDuration,
            trackCount: trackConfigs.length,
            source: 'refactored-v2'
        }
    };
}

/**
 * Generate executable Tone.js code (replaces old generator template)
 */
function generateExecutableCode(CDN_BASE, tempo, trackConfigs, trackSchedules, maxDuration) {
    const configsJSON = JSON.stringify(trackConfigs);
    const schedulesJSON = JSON.stringify(trackSchedules);

    return `
// Auto-generated Tone.js playback code
(async function() {
    console.log('[Music] Initializing playback...');
    
    const CDN_BASE = '${CDN_BASE}';
    const tempo = ${tempo};
    const duration = ${maxDuration};
    const trackConfigs = ${configsJSON};
    const trackSchedules = ${schedulesJSON};
    
    // PERSISTENT CACHE: Store in window so it survives between plays
    if (!window.__musicCache) {
        window.__musicCache = {
            mappings: new Map(),
            instrumentPools: new Map()
        };
        console.log('[Music] Created persistent cache');
    }
    
    const mappingCache = window.__musicCache.mappings;
    const instrumentPools = window.__musicCache.instrumentPools;
    
    /**
     * Analyze sample loudness and calculate needed gain
     */
    async function analyzeInstrumentGain(urls, baseUrl) {
        try {
            const sampleKeys = Object.keys(urls).slice(0, Math.min(3, Object.keys(urls).length));
            if (sampleKeys.length === 0) return 0;

            const audioContext = Tone.context.rawContext;
            const peakLevels = [];

            for (const key of sampleKeys) {
                try {
                    const sampleUrl = baseUrl + urls[key];
                    const response = await fetch(sampleUrl);
                    const arrayBuffer = await response.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                    let maxPeak = 0;
                    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
                        const channelData = audioBuffer.getChannelData(channel);
                        for (let i = 0; i < channelData.length; i++) {
                            maxPeak = Math.max(maxPeak, Math.abs(channelData[i]));
                        }
                    }

                    peakLevels.push(maxPeak);
                    console.log(\`[Gain Analysis] Sample \${key}: peak = \${maxPeak.toFixed(6)} (\${(20 * Math.log10(maxPeak)).toFixed(1)}dB)\`);
                } catch (err) {
                    console.warn(\`[Gain Analysis] Failed to analyze \${key}:\`, err.message);
                }
            }

            if (peakLevels.length === 0) return 0;

            const avgPeak = peakLevels.reduce((sum, p) => sum + p, 0) / peakLevels.length;
            const avgPeakDb = 20 * Math.log10(avgPeak);
            const targetDb = -6;
            const neededGain = targetDb - avgPeakDb;

            console.log(\`[Gain Analysis] Average peak: \${avgPeak.toFixed(6)} (\${avgPeakDb.toFixed(1)}dB)\`);
            console.log(\`[Gain Analysis] Recommended gain: \${neededGain.toFixed(1)}dB\`);

            return Math.max(-20, Math.min(60, neededGain));
        } catch (error) {
            console.error('[Gain Analysis] Error:', error);
            return 0;
        }
    }
    
    // Load instrument mapping
    async function loadMapping(instrumentPath) {
        if (mappingCache.has(instrumentPath)) {
            console.log('[Music] Using cached mapping:', instrumentPath);
            return mappingCache.get(instrumentPath);
        }
        
        const url = CDN_BASE + '/samples/' + instrumentPath + '/mapping.json';
        console.log('[Music] Fetching mapping:', url);
        
        try {
            const response = await fetch(url, { cache: 'force-cache' });
            
            if (!response.ok) {
                throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
            }
            
            const mapping = await response.json();
            
            if (!mapping) {
                throw new Error(\`Invalid mapping structure - empty response\`);
            }
            
            const noteCount = mapping.velocity_layers 
                ? Object.keys(mapping.velocity_layers).length 
                : (mapping.samples ? Object.keys(mapping.samples).length : 0);
            
            const format = mapping.velocity_layers ? 'velocity layers' : 'simple samples';
            console.log('[Music] Loaded mapping for:', instrumentPath, '- Notes:', noteCount, '- Format:', format);
            
            mappingCache.set(instrumentPath, mapping);
            return mapping;
        } catch (error) {
            console.error('[Music] ERROR loading mapping for:', instrumentPath);
            console.error('[Music] URL was:', url);
            console.error('[Music] Error:', error);
            throw error;
        }
    }
    
    // Build sampler URLs from mapping
    function buildSamplerUrls(mapping, instrumentPath) {
        const urls = {};
        const baseUrl = CDN_BASE + '/samples/' + instrumentPath + '/';
        
        if (mapping.velocity_layers) {
            for (const [note, layers] of Object.entries(mapping.velocity_layers)) {
                let sampleKey = note;
                
                if (mapping.type === 'drums') {
                    const drumNoteMap = {
                        "kick": "C2", "snare": "D2", "snare_rimshot": "E2",
                        "snare_buzz": "D#2", "hihat_closed": "F#2", "tom": "G2",
                        "crash": "C#3", "ride": "D#3", "hihat_pedal": "G#2", "hihat_open": "A#2"
                    };
                    sampleKey = drumNoteMap[note] || note;
                }
                
                let selectedLayer = layers.find(l => 
                    l.file.includes('Sustains') || l.file.includes('sus')
                ) || layers.find(l => l.file.includes('vel4')) || layers[Math.floor(layers.length / 2)];
                
                if (selectedLayer) {
                    urls[sampleKey] = selectedLayer.file.split('/').map(encodeURIComponent).join('/');
                }
            }
        }
        else if (mapping.samples) {
            for (const [note, file] of Object.entries(mapping.samples)) {
                urls[note] = file.split('/').map(encodeURIComponent).join('/');
            }
        }
        
        return { urls, baseUrl };
    }
    
    async function createInstrumentPool(instrumentName, voiceCount = 8) {
        const synthFactories = {
            'bass': () => new Tone.MonoSynth({
                oscillator: { type: 'square' },
                envelope: { attack: 0.01, decay: 0.2, sustain: 0.4, release: 0.3 },
                filter: { type: 'lowpass', frequency: 800 }
            }).toDestination(),
            
            'synth_lead': () => new Tone.MonoSynth({
                oscillator: { type: 'sawtooth' },
                envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 1 },
                filterEnvelope: { 
                    attack: 0.06, decay: 0.2, sustain: 0.5, release: 2, 
                    baseFrequency: 200, octaves: 7 
                }
            }).toDestination(),
            
            'pad': () => new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: 'sine' },
                envelope: { attack: 0.5, decay: 0.3, sustain: 0.7, release: 2 }
            }).toDestination(),
            
            'bells': () => new Tone.MetalSynth({
                frequency: 200,
                envelope: { attack: 0.001, decay: 1.4, release: 0.2 },
                harmonicity: 5.1,
                modulationIndex: 32,
                resonance: 4000
            }).toDestination(),
            
            'strings': () => new Tone.PolySynth(Tone.Synth, {
                oscillator: { type: 'sawtooth' },
                envelope: { attack: 0.2, decay: 0.1, sustain: 0.8, release: 1.5 }
            }).toDestination()
        };
        
        if (synthFactories[instrumentName]) {
            const pool = [];
            for (let i = 0; i < voiceCount; i++) {
                pool.push(synthFactories[instrumentName]());
            }
            
            return {
                voices: pool,
                currentVoice: 0,
                play(note, duration, time, velocity) {
                    const voice = this.voices[this.currentVoice];
                    this.currentVoice = (this.currentVoice + 1) % this.voices.length;
                    voice.triggerAttackRelease(note, duration, time, velocity);
                },
                playChord(notes, duration, time, velocity) {
                    notes.forEach(note => this.play(note, duration, time, velocity));
                }
            };
        }
        
        const mapping = await loadMapping(instrumentName);
        const { urls, baseUrl } = buildSamplerUrls(mapping, instrumentName);
        
        console.log(\`[Music] Analyzing gain for \${instrumentName}...\`);
        const calculatedGain = await analyzeInstrumentGain(urls, baseUrl);
        console.log(\`[Music] Applying \${calculatedGain.toFixed(1)}dB gain to \${instrumentName}\`);
        
        const pool = [];
        
        for (let i = 0; i < voiceCount; i++) {
            const sampler = await new Promise((resolve, reject) => {
                const s = new Tone.Sampler({
                    urls,
                    baseUrl,
                    volume: calculatedGain,
                    onload: () => resolve(s),
                    onerror: reject
                }).toDestination();
            });
            sampler.__preGain = calculatedGain;
            pool.push(sampler);
        }
        
        return {
            voices: pool,
            currentVoice: 0,
            play(note, duration, time, velocity) {
                const voice = this.voices[this.currentVoice];
                this.currentVoice = (this.currentVoice + 1) % this.voices.length;
                voice.triggerAttackRelease(note, duration, time, velocity);
            },
            playChord(notes, duration, time, velocity) {
                notes.forEach(note => this.play(note, duration, time, velocity));
            }
        };
    }
    
    // Load all instruments
    console.log('[Music] Loading instruments...');
    const loadPromises =[];
    
    for (const config of trackConfigs) {
        if (!instrumentPools.has(config.trackId)) {
            const loadPromise = createInstrumentPool(config.instrumentName, 8)
                .then(pool => {
                    instrumentPools.set(config.trackId, pool);
                    console.log('[Music] Loaded:', config.instrumentName);
                });
            loadPromises.push(loadPromise);
        } else {
            console.log('[Music] Reusing cached pool:', config.instrumentName);
        }
    }
    
    await Promise.all(loadPromises);
    console.log('[Music] All instruments ready');
    
    if (Tone.context.state !== 'running') {
        await Tone.start();
        console.log('[Music] Audio context started');
    }
    
    await new Promise(resolve => setTimeout(resolve, 100));
    
    Tone.Transport.cancel();
    Tone.Transport.bpm.value = tempo;
    
    console.log('[Music] Scheduling events...');
    for (const schedule of trackSchedules) {
        const pool = instrumentPools.get(schedule.trackId);
        if (!pool) continue;
        
        schedule.notes.forEach(noteData => {
            Tone.Transport.schedule((time) => {
                pool.play(noteData.note, noteData.duration, time, noteData.velocity);
            }, noteData.time);
        });
        
        schedule.chords.forEach(chordData => {
            Tone.Transport.schedule((time) => {
                pool.playChord(chordData.notes, chordData.duration, time, chordData.velocity);
            }, chordData.time);
        });
    }
    
    Tone.Transport.start();
    console.log('[Music] Playing... (' + duration.toFixed(2) + 's)');
    
    const playbackTimeout = setTimeout(() => {
        Tone.Transport.stop();
        console.log('[Music] Playback complete');
    }, (duration + 1) * 1000);
    
    window.__musicControls = {
        stop: () => {
            clearTimeout(playbackTimeout);  // Clear the auto-stop timeout
            Tone.Transport.stop();
            console.log('[Music] Playback complete');
        },
        pause: () => Tone.Transport.pause(),
        resume: () => Tone.Transport.start(),
        pools: instrumentPools
    };
})();
`;
}

/**
 * Helper: MIDI to note name
 */
function midiToNote(midi) {
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(midi / 12) - 1;
    const note = notes[midi % 12];
    return note + octave;
}

// Error handling
app.use((err, req, res, next) => {
    console.error('[ERROR]', err);
    res.status(500).json({
        status: 'error',
        message: err.message
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════╗
║   Music Runner v2.0                    ║
║   Port: ${PORT}                            ║
║   CDN: Cloudflare R2                   ║
╚════════════════════════════════════════╝
    `);
    console.log('Endpoints:');
    console.log('  POST /eval - Compile DSL/IR to Tone.js');
    console.log('  GET  /health - Health check');
    console.log('');
    console.log('Sample directory ready');
    console.log('Client modules available at /client/');
});