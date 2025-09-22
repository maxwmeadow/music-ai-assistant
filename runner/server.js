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
    const { musicData } = req.body;

    if (!musicData) {
        return res.status(400).json({ status: 'error', message: 'musicData required' });
    }

    try {
        const parser = new MusicJSONParser();
        const parsedData = parser.parse(JSON.stringify(musicData));

        const generator = new DSLGenerator();
        const dslCode = generator.generate(parsedData);
        const executableCode = generator.compileDSLToToneJS(dslCode);

        res.json({
            status: 'success',
            dsl_code: dslCode,
            executable_code: executableCode,
            parsed_data: parsedData
        });
    } catch (error) {
        console.error('[EVAL] Error:', error);
        res.status(500).json({ status: 'error', message: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`[RUNNER] Server running on port ${PORT}`);
});