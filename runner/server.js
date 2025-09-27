const express = require('express');
const cors = require('cors');
const MusicJSONParser = require('./parser');
const DSLGenerator = require('./generator');

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.path}`);
    console.log(`[DEBUG] Body:`, req.body);
    next();
});

app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'music-runner' });
});

app.post('/eval', (req, res) => {
    console.log('[DEBUG] Eval endpoint hit');
    const { musicData } = req.body;

    if (!musicData) {
        console.log('[DEBUG] No musicData provided');
        return res.status(400).json({ status: 'error', message: 'musicData required' });
    }

    console.log('[DEBUG] musicData received:', JSON.stringify(musicData, null, 2));

    try {
        let dslCode;
        let parsedData;

        // Check if this is a DSL passthrough
        if (musicData.__dsl_passthrough) {
            console.log('[DEBUG] DSL passthrough detected');
            dslCode = musicData.__dsl_passthrough;
            console.log('[DEBUG] DSL code to process:', dslCode);

            // Don't use dummy data - let the generator work with the actual DSL
            parsedData = {
                metadata: { tempo: 120 },
                tracks: [],
                __source: "dsl_passthrough"
            };
        } else {
            // Normal IR processing
            console.log('[DEBUG] Processing as IR data');
            const parser = new MusicJSONParser();
            parsedData = parser.parse(JSON.stringify(musicData));
            console.log('[DEBUG] Parsed IR data:', parsedData);

            const generator = new DSLGenerator();
            dslCode = generator.generate(parsedData);
            console.log('[DEBUG] Generated DSL from IR:', dslCode);
        }

        console.log('[DEBUG] About to compile DSL to Tone.js');
        console.log('[DEBUG] Final DSL code:', dslCode);

        const generator = new DSLGenerator();
        const executableCode = generator.compileDSLToToneJS(dslCode);

        console.log('[DEBUG] Generated executable code length:', executableCode.length);
        console.log('[DEBUG] Executable code preview:', executableCode.substring(0, 200) + '...');

        const response = {
            status: 'success',
            dsl_code: dslCode,
            executable_code: executableCode,
            parsed_data: parsedData
        };

        console.log('[DEBUG] Sending response');
        res.json(response);
    } catch (error) {
        console.error('[EVAL] Error:', error);
        console.error('[EVAL] Stack:', error.stack);
        res.status(500).json({ status: 'error', message: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`[RUNNER] Server running on port ${PORT}`);
});