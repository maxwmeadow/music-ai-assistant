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

            // Check if there's also IR data (for audio clips)
            if (musicData.tracks && musicData.tracks.length > 0) {
                console.log('[EVAL] IR data also present (audio clips)');
                const parser = new MusicJSONParser();
                irData = parser.parse(JSON.stringify(musicData));
            } else {
                irData = null;
            }
        } else {
            console.log('[EVAL] IR mode - converting to DSL');
            // Convert IR to DSL
            const parser = new MusicJSONParser();
            irData = parser.parse(JSON.stringify(musicData));
            dslCode = irToDSL(irData);
        }

        console.log('[EVAL] DSL Code:', dslCode.substring(0, 200) + '...');

        // Compile DSL + IR to executable Tone.js code
        const result = await compileDSLToExecutable(dslCode, irData);

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

        // Check if this is a drum track
        const isDrumTrack = track.id.toLowerCase().includes('drum') ||
                           track.instrument?.toLowerCase().includes('drum');

        if (track.notes) {
            track.notes.forEach(note => {
                // Use drum names for drum tracks, regular notes for others
                const noteName = isDrumTrack ? midiToDrumName(note.pitch) : midiToNote(note.pitch);
                dsl += `  note("${noteName}", ${note.start}, ${note.duration}, ${note.velocity})\n`;
            });
        }

        if (track.samples) {
            track.samples.forEach(sample => {
                // Convert samples to note() format for consistency
                const duration = sample.duration || 0.5;
                const velocity = sample.velocity || 0.8;
                dsl += `  note("${sample.sample}", ${sample.start}, ${duration}, ${velocity})\n`;
            });
        }

        dsl += `}\n\n`;
    });

    return dsl;
}

/**
 * Expand loop constructs in DSL code
 * Supports: loop(startTime, endTime) { note(pitch, relativeTime, duration, velocity) }
 */
function expandLoops(dslCode) {
    let expandedCode = dslCode;
    let maxIterations = 100;
    let iteration = 0;

    // Time-based loop pattern: loop(startTime, endTime) { content }
    const loopPattern = /loop\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g;
    let match;

    while ((match = loopPattern.exec(expandedCode)) !== null && iteration < maxIterations) {
        const fullMatch = match[0];
        const startTime = parseFloat(match[1]);
        const endTime = parseFloat(match[2]);
        const loopContent = match[3];

        // Parse notes to find pattern duration
        const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
        const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;

        let maxTime = 0;
        let noteMatch;

        // Find max time in notes
        while ((noteMatch = notePattern.exec(loopContent)) !== null) {
            const relativeStart = parseFloat(noteMatch[2]);
            const duration = parseFloat(noteMatch[3]);
            maxTime = Math.max(maxTime, relativeStart + duration);
        }

        // Find max time in chords
        while ((noteMatch = chordPattern.exec(loopContent)) !== null) {
            const relativeStart = parseFloat(noteMatch[2]);
            const duration = parseFloat(noteMatch[3]);
            maxTime = Math.max(maxTime, relativeStart + duration);
        }

        if (maxTime === 0) {
            // No notes found, skip this loop
            expandedCode = expandedCode.replace(fullMatch, '');
            loopPattern.lastIndex = 0;
            iteration++;
            continue;
        }

        const patternDuration = maxTime;
        const loopDuration = endTime - startTime;
        const repetitions = Math.ceil(loopDuration / patternDuration);

        let expandedContent = '';

        // Generate repeated notes with absolute timing
        for (let rep = 0; rep < repetitions; rep++) {
            const repStartTime = startTime + (rep * patternDuration);

            if (repStartTime >= endTime) break;

            // Reset regex for this repetition
            notePattern.lastIndex = 0;
            chordPattern.lastIndex = 0;

            // Expand notes
            while ((noteMatch = notePattern.exec(loopContent)) !== null) {
                const pitch = noteMatch[1];
                const relativeStart = parseFloat(noteMatch[2]);
                const duration = parseFloat(noteMatch[3]);
                const velocity = parseFloat(noteMatch[4]);

                const absoluteStart = repStartTime + relativeStart;

                if (absoluteStart < endTime) {
                    expandedContent += `    note("${pitch}", ${absoluteStart.toFixed(4)}, ${duration}, ${velocity})\n`;
                }
            }

            // Expand chords
            while ((noteMatch = chordPattern.exec(loopContent)) !== null) {
                const notes = noteMatch[1];
                const relativeStart = parseFloat(noteMatch[2]);
                const duration = parseFloat(noteMatch[3]);
                const velocity = parseFloat(noteMatch[4]);

                const absoluteStart = repStartTime + relativeStart;

                if (absoluteStart < endTime) {
                    expandedContent += `    chord([${notes}], ${absoluteStart.toFixed(4)}, ${duration}, ${velocity})\n`;
                }
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
 * Extract track blocks from DSL with proper brace matching
 */
function extractTracks(dslCode) {
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
 * Convert flat note names to their sharp equivalents
 * This allows users to write either Bb4 or A#4 - both will work
 * @param {string} dslCode - The DSL code that may contain flat notes
 * @returns {string} DSL code with all flats converted to sharps
 */
function convertFlatsToSharps(dslCode) {
    // Mapping of flats to sharps
    const flatToSharp = {
        'Bb': 'A#',
        'Eb': 'D#',
        'Ab': 'G#',
        'Db': 'C#',
        'Gb': 'F#'
    };

    // Replace all quoted flat notes (e.g., "Bb4", "Eb3")
    // Pattern matches: "NoteOctave" where Note is a letter with 'b'
    let converted = dslCode;

    for (const [flat, sharp] of Object.entries(flatToSharp)) {
        // Match quoted notes like "Bb4" and replace with "A#4"
        // Use regex with global flag to replace all occurrences
        const regex = new RegExp(`"${flat}(\\d+)"`, 'g');
        converted = converted.replace(regex, `"${sharp}$1"`);
    }

    return converted;
}

/**
 * Compile DSL to executable Tone.js code
 * This is the core replacement for generator.js
 * @param {string} dslCode - The DSL code to compile
 * @param {object|null} irData - Optional IR data (for audio clips)
 */
async function compileDSLToExecutable(dslCode, irData = null) {
    const CDN_BASE = 'https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev';

    // Convert flats to sharps (Bb→A#, Eb→D#, etc.)
    dslCode = convertFlatsToSharps(dslCode);
    console.log('[EVAL] Converted flats to sharps');

    // Expand loops before parsing
    dslCode = expandLoops(dslCode);
    console.log('[EVAL] Expanded DSL length:', dslCode.length);

    // Parse DSL
    const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
    const tempo = tempoMatch ? parseInt(tempoMatch[1]) : 120;

    const trackMatches = extractTracks(dslCode);
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

        // Check if this is a drum track
        const isDrumTrack = trackId.toLowerCase().includes('drum') ||
                           instrumentName?.toLowerCase().includes('drum');

        // Parse notes (4-parameter with absolute timing)
        const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
        const notes = [];

        if (noteMatches) {
            noteMatches.forEach(noteMatch => {
                const [, noteName, start, duration, velocity] =
                    noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);

                // Convert drum names to MIDI notes for drum tracks
                const finalNote = isDrumTrack ? drumNameToMidiNote(noteName) : noteName;

                notes.push({
                    note: finalNote,
                    duration: parseFloat(duration),
                    velocity: parseFloat(velocity),
                    time: parseFloat(start)
                });
            });
        }

        // Parse chords (4-parameter with absolute timing)
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
                    time: parseFloat(start)
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

    // Add audio clips from IR data if present
    const audioClips = [];
    if (irData && irData.tracks) {
        irData.tracks.forEach(track => {
            if (track.audio) {
                track.audio.forEach(clip => {
                    audioClips.push({
                        trackId: track.id,
                        audioData: clip.audio_data,
                        start: clip.start,
                        duration: clip.duration,
                        volume: clip.volume || 1.0
                    });

                    // Update max duration to include audio clips
                    const clipEnd = clip.start + clip.duration;
                    maxDuration = Math.max(maxDuration, clipEnd);
                });
            }
        });
    }

    maxDuration += 1;

    // Generate standalone executable code
    const executable_code = generateExecutableCode(
        CDN_BASE,
        tempo,
        trackConfigs,
        trackSchedules,
        audioClips,
        maxDuration
    );

    return {
        dsl_code: dslCode,
        executable_code,
        parsed_data: {
            tempo,
            duration: maxDuration,
            trackCount: trackConfigs.length,
            audioClipCount: audioClips.length,
            source: 'refactored-v2'
        }
    };
}

/**
 * Generate executable Tone.js code (replaces old generator template)
 */
function generateExecutableCode(CDN_BASE, tempo, trackConfigs, trackSchedules, audioClips, maxDuration) {
    const configsJSON = JSON.stringify(trackConfigs);
    const schedulesJSON = JSON.stringify(trackSchedules);
    const audioClipsJSON = JSON.stringify(audioClips);

    return `
// Auto-generated Tone.js playback code
(async function() {
    console.log('[Music] Initializing playback...');
    
    const CDN_BASE = '${CDN_BASE}';
    const tempo = ${tempo};
    const duration = ${maxDuration};
    const trackConfigs = ${configsJSON};
    const trackSchedules = ${schedulesJSON};
    const audioClips = ${audioClipsJSON};

    // Pre-calculated gain compensation map (in dB)
    // Target level: -12dB peak
    const INSTRUMENT_GAINS = {
        // Pianos
        'piano/steinway_grand': 0,
        'piano/bechstein_1911_upright': 0,
        'piano/fender_rhodes': 0,
        'piano/experience_ny_steinway': 0,
        // Harpsichords
        'harpsichord/harpsichord_english': -10,
        'harpsichord/harpsichord_flemish': -10,
        'harpsichord/harpsichord_french': -10,
        'harpsichord/harpsichord_italian': -15,
        'harpsichord/harpsichord_unk': -10,
        // Guitars
        'guitar/rjs_guitar_palm_muted_softly_strings': -5,
        'guitar/rjs_guitar_palm_muted_strings': -6,
        'synth/lead/ld_the_stack_guitar_chug': 0,
        'synth/lead/ld_the_stack_guitar': -1,
        'guitar/rjs_guitar_new_strings': 0,
        'guitar/rjs_guitar_old_strings': 4,
        // Bass
        'bass/funky_fingers': -10,
        'bass/low_fat_bass': -5,
        'bass/jp8000_sawbass': 2,
        'bass/jp8000_tribass': 2,
        // Strings
        'strings/nfo_chamber_strings_longs': 0,
        'strings/nfo_iso_celli_swells': 0,
        'strings/nfo_iso_viola_swells': 0,
        'strings/nfo_iso_violin_swells': 0,
        // Brass
        'brass/nfo_iso_brass_swells': 0,
        // Winds
        'winds/flute_violin': 0,
        'winds/subtle_clarinet': 0,
        'winds/decent_oboe': 0,
        'winds/tenor_saxophone': 0,
    };

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

        // Helper to create URL with fallback extensions
        const normalizeFileUrl = (file) => {
            // Try lowercase .wav first, store uppercase .WAV as fallback
            const encodedPath = file.split('/').map(encodeURIComponent).join('/');
            return {
                primary: encodedPath.replace(/\.WAV$/i, '.wav'),
                fallback: encodedPath.replace(/\.wav$/i, '.WAV')
            };
        };

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
                    const { primary, fallback } = normalizeFileUrl(selectedLayer.file);
                    urls[sampleKey] = { primary, fallback };
                }
            }
        }
        else if (mapping.samples) {
            for (const [note, file] of Object.entries(mapping.samples)) {
                const { primary, fallback } = normalizeFileUrl(file);
                urls[note] = { primary, fallback };
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

        // Convert urls object to primary URLs first
        const primaryUrls = {};
        const fallbackUrls = {};
        for (const [note, urlObj] of Object.entries(urls)) {
            if (typeof urlObj === 'object' && urlObj.primary) {
                primaryUrls[note] = urlObj.primary;
                fallbackUrls[note] = urlObj.fallback;
            } else {
                primaryUrls[note] = urlObj;
                fallbackUrls[note] = urlObj;
            }
        }

        // Use pre-calculated gain if available, otherwise analyze dynamically
        let calculatedGain;
        if (INSTRUMENT_GAINS.hasOwnProperty(instrumentName)) {
            calculatedGain = INSTRUMENT_GAINS[instrumentName];
            console.log(\`[Music] Using pre-calculated gain for \${instrumentName}: \${calculatedGain}dB\`);
        } else {
            console.log(\`[Music] Analyzing gain for \${instrumentName}...\`);
            calculatedGain = await analyzeInstrumentGain(primaryUrls, baseUrl);
            console.log(\`[Music] Applying \${calculatedGain.toFixed(1)}dB gain to \${instrumentName}\`);
        }

        const pool = [];

        for (let i = 0; i < voiceCount; i++) {
            let sampler;
            try {
                // Try primary URLs first (.wav)
                sampler = await new Promise((resolve, reject) => {
                    const s = new Tone.Sampler({
                        urls: primaryUrls,
                        baseUrl,
                        volume: calculatedGain,
                        onload: () => resolve(s),
                        onerror: reject
                    }).toDestination();
                });
            } catch (error) {
                console.log(\`[Music] Primary URLs failed for \${instrumentName}, trying fallback (.WAV)...\`);
                // Fallback to .WAV extension
                sampler = await new Promise((resolve, reject) => {
                    const s = new Tone.Sampler({
                        urls: fallbackUrls,
                        baseUrl,
                        volume: calculatedGain,
                        onload: () => resolve(s),
                        onerror: reject
                    }).toDestination();
                });
            }
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
        // Use composite key: trackId + instrumentName to detect instrument changes
        const cacheKey = config.trackId + '::' + config.instrumentName;
        const existingPool = instrumentPools.get(config.trackId);
        const existingCacheKey = existingPool?.__cacheKey;

        // Only reuse if the same instrument is cached for this track
        if (existingPool && existingCacheKey === cacheKey) {
            console.log('[Music] Reusing cached pool:', config.instrumentName);
        } else {
            if (existingPool) {
                console.log('[Music] Instrument changed for track', config.trackId, '- reloading');
            }
            const loadPromise = createInstrumentPool(config.instrumentName, 8)
                .then(pool => {
                    pool.__cacheKey = cacheKey;
                    instrumentPools.set(config.trackId, pool);
                    console.log('[Music] Loaded:', config.instrumentName);
                });
            loadPromises.push(loadPromise);
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

    // Load and schedule audio clips (vocal recordings)
    console.log('[Music] Loading ' + audioClips.length + ' audio clips...');
    const audioPlayers = [];

    for (let i = 0; i < audioClips.length; i++) {
        const clip = audioClips[i];
        const audioUrl = 'data:audio/wav;base64,' + clip.audioData;

        console.log('[Music] Creating player for clip ' + i + ', audio size: ' + clip.audioData.length + ' chars');

        const player = new Tone.Player({
            url: audioUrl,
            volume: Tone.gainToDb(clip.volume)
        }).toDestination();

        // Wait for player to load (with error handling)
        try {
            await new Promise((resolve, reject) => {
                let loaded = false;

                player.onload = () => {
                    if (!loaded) {
                        loaded = true;
                        console.log('[Music] Audio clip ' + i + ' loaded for track ' + clip.trackId + ' (duration: ' + player.buffer.duration.toFixed(2) + 's)');
                        resolve(true);
                    }
                };

                player.onerror = (error) => {
                    if (!loaded) {
                        loaded = true;
                        console.error('[Music] Failed to load audio clip ' + i + ':', error);
                        reject(error);
                    }
                };

                // Timeout after 10 seconds
                setTimeout(() => {
                    if (!loaded) {
                        loaded = true;
                        console.error('[Music] Audio clip ' + i + ' load timeout');
                        reject(new Error('Audio clip load timeout'));
                    }
                }, 10000);
            });

            audioPlayers.push({ player, clip, index: i });
        } catch (error) {
            console.error('[Music] Skipping audio clip ' + i + ' due to load error:', error);
            // Continue without this clip instead of failing completely
        }
    }

    console.log('[Music] Successfully loaded ' + audioPlayers.length + ' of ' + audioClips.length + ' audio clips');

    // Schedule all loaded audio clips
    audioPlayers.forEach(({ player, clip, index }) => {
        Tone.Transport.schedule((time) => {
            try {
                player.start(time);
                console.log('[Music] Playing audio clip ' + index + ' at ' + clip.start + 's');
            } catch (error) {
                console.error('[Music] Error playing audio clip ' + index + ':', error);
            }
        }, clip.start);
    });

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

/**
 * Convert MIDI number to drum name for drum tracks
 */
function midiToDrumName(midi) {
    const drumMap = {
        36: 'kick',          // C2
        38: 'snare',         // D2
        40: 'snare_rimshot', // E2
        39: 'snare_buzz',    // D#2
        42: 'hihat_closed',  // F#2
        46: 'hihat_open',    // A#2
        44: 'hihat_pedal',   // G#2
        43: 'tom',           // G2
        49: 'crash',         // C#3
        51: 'ride',          // D#3
    };

    return drumMap[midi] || midiToNote(midi);
}

/**
 * Convert drum name to MIDI note for playback
 */
function drumNameToMidiNote(noteName) {
    const drumNameMap = {
        'kick': 'C2',
        'snare': 'D2',
        'snare_rimshot': 'E2',
        'snare_buzz': 'D#2',
        'hihat_closed': 'F#2',
        'hihat_open': 'A#2',
        'hihat_pedal': 'G#2',
        'tom': 'G2',
        'crash': 'C#3',
        'ride': 'D#3',
        // Aliases
        'hihat': 'F#2',  // Default to closed
    };

    const lowerName = noteName.toLowerCase();
    return drumNameMap[lowerName] || noteName; // Return as-is if not a drum name
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