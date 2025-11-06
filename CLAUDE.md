# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Music AI Assistant is a web-based music generation system with AI-powered audio-to-music conversion. The system converts humming to melody and beatboxing to drum patterns using trained deep learning models, then renders them as playable music through a browser-based DSL compiler and audio engine.

## Architecture

### Three-Service Architecture

1. **Backend** (FastAPI, Python) - Port 8000
   - ML model inference for hum2melody and beatbox2drums
   - Audio preprocessing with librosa
   - Training data collection (SQLite)
   - IR (Intermediate Representation) management

2. **Runner** (Express, Node.js) - Port 5001
   - Compiles music DSL to executable Tone.js code
   - Sample caching and CDN management
   - Real-time audio synthesis

3. **Frontend** (Next.js 15, React 19, TypeScript) - Port 3000
   - Monaco editor for DSL editing
   - Visual timeline editor (piano-roll)
   - Audio mixer with per-track controls
   - Tone.js playback engine

### Key Flow

```
User Audio → Backend (ML inference) → IR → Runner (DSL compilation) → Frontend (Tone.js playback)
```

## Development Commands

### Starting Services

**All services (Windows):**
```powershell
.\start-all.ps1
```

**Individual services:**
```bash
# Runner
cd runner && npm start

# Backend (activate venv first)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm run dev
```

### Backend Virtual Environment

The backend uses a Python virtual environment located at `backend/.venv`. Always activate it before running backend commands:

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Python Dependencies

Install with:
```bash
cd backend
pip install -r requirements.txt
```

Key dependencies: `torch`, `librosa`, `fastapi`, `uvicorn`, `numpy`, `soundfile`

### Frontend Commands

```bash
cd frontend
npm run dev      # Development server (port 3000)
npm run build    # Production build
npm run start    # Production server
npm run lint     # ESLint
```

### Testing

```bash
# Test backend endpoints
python backend/test_audio_endpoints.py

# Test model and dataset
python backend/test_model_dataset.py
```

## Model System

### Trained Models

The system uses a combined PyTorch model stored in `backend/checkpoints/`:
- `combined_hum2melody_full.pth` - Production hum2melody model (135MB)
  - Combines pitch detection (Hum2MelodyModel) and onset/offset detection (OnsetOffsetModel)
  - 4 output heads: frame (pitch), onset, offset, f0 (continuous pitch + voicing)
  - ~18.6M total parameters

### Model Loading Flow

1. `backend/main.py` initializes `ModelServer` on startup
2. `ModelServer.__init__()` attempts to load `backend.inference.predictor.MelodyPredictor`
3. Predictor loads combined checkpoint from `backend/checkpoints/combined_hum2melody_full.pth`
4. Combined model loader reconstructs both pitch and onset models
5. If model loading fails, falls back to mock predictions

### Model Architecture

- **Combined model**: `backend/models/combined_model_loader.py` - `CombinedModelFromCheckpoint`
  - **Pitch model**: `backend/models/hum2melody_model.py` - `Hum2MelodyModel` (~14.9M params)
  - **Onset model**: `backend/models/onset_model.py` - `OnsetOffsetModel` (~3.7M params)
- **Inference wrapper**: `backend/inference/predictor.py` - `MelodyPredictor`
- **Musical components**: `backend/models/musical_components.py` - Support classes
- **Dataset**: `backend/data/melody_dataset.py` - `MelodyDataset` (CQT-based)

### Key Model Parameters

```python
sample_rate = 16000
n_bins = 88  # CQT bins for MIDI 21-108
bins_per_octave = 12
fmin = 27.5  # A0
target_frames = 500
hop_length = 512
min_midi = 21
max_midi = 108
num_notes = 88
frame_rate = 7.8125  # fps after 4x CNN downsampling
```

## IR (Intermediate Representation)

Defined in `backend/schemas.py`:

```python
IR = {
    "metadata": {"tempo": int, "key": str, ...},
    "tracks": [Track]
}

Track = {
    "id": str,
    "instrument": str | None,
    "notes": [Note] | None,
    "samples": [SampleEvent] | None
}

Note = {
    "pitch": int,        # MIDI number
    "start": float,      # seconds
    "duration": float,   # seconds
    "velocity": float    # 0.0-1.0
}
```

## API Endpoints

### Backend (port 8000)

- `GET /health` - Health check
- `GET /stats` - Database statistics
- `POST /hum2melody` - Convert humming audio to melody
  - Form data: `audio` (file), `save_training_data` (bool), `instrument` (str)
  - Returns: IR with melody track
- `POST /beatbox2drums` - Convert beatbox audio to drums
  - Form data: `audio` (file), `save_training_data` (bool)
  - Returns: IR with drum track
- `POST /arrange` - Add accompaniment tracks to existing IR
  - Body: `{"ir": IR, "style": "pop"|"jazz"|"electronic"}`
- `POST /feedback` - Submit user feedback on predictions
- `POST /run` - Forward DSL/IR to runner for compilation

### Runner (port 5001)

- `GET /health` - Health check
- `POST /eval` - Compile IR/DSL to executable Tone.js code
  - Body: `{"ir": IR}` or raw DSL string in `__dsl_passthrough`
  - Returns: Compiled JavaScript code

## Database

SQLite database at `backend/training_data.db` stores:
- Audio samples with metadata
- Model predictions
- User feedback (1-5 ratings)

Access via `TrainingDataDB` class in `backend/database.py`.

## Training Data Collection

All audio uploaded to `/hum2melody` and `/beatbox2drums` is automatically saved to:
- File: `backend/audio_uploads/`
- Database: `backend/training_data.db`

This enables future model retraining with real user data.

## Important Implementation Notes

### Audio Processing

1. **Preprocessing**: Audio is converted to CQT (Constant-Q Transform) spectrograms with 88 bins covering MIDI range 21-108
2. **Sample rate**: All audio is resampled to 16kHz for model inference
3. **Format**: Models expect mono audio
4. **Frame rate**: Model outputs at ~7.8 fps after 4x CNN downsampling (original hop_length=512 gives 31.25 fps)

### Model Inference Modes

The predictor has two modes:
- `predict_from_audio()` - Standard with post-processing (note merging, overlap resolution)
- `predict_from_audio_RAW()` - Minimal post-processing, more direct model output

### Frontend-Backend Integration

1. Frontend sends audio via multipart/form-data
2. Backend preprocesses with `AudioProcessor`
3. Model inference via `ModelServer.predict_melody()` or `predict_drums()`
4. Returns IR to frontend
5. Frontend forwards IR to runner via `/run` endpoint
6. Runner compiles to Tone.js and returns executable code
7. Frontend executes code for playback

### DSL Format

Music is represented in a custom DSL:

```javascript
tempo(120)

track("melody") {
  instrument("piano/grand_piano_k")
  note("C4", 0.5, 0.8)  // pitch, duration, velocity
  chord(["C4", "E4", "G4"], 2.0, 0.6)
}
```

## File Structure

```
music-ai-assistant/
├── backend/
│   ├── main.py                 # FastAPI app with ML endpoints
│   ├── model_server.py         # Model loading and inference
│   ├── audio_processor.py      # Audio preprocessing (librosa)
│   ├── database.py             # Training data storage
│   ├── schemas.py              # Pydantic models (IR, Track, Note)
│   ├── inference/
│   │   └── predictor.py        # MelodyPredictor inference wrapper
│   ├── models/
│   │   ├── hum2melody_model.py        # Hum2MelodyModel (pitch detection)
│   │   ├── onset_model.py             # OnsetOffsetModel
│   │   ├── combined_model.py          # CombinedHum2MelodyModel wrapper
│   │   ├── combined_model_loader.py   # Single-file checkpoint loader
│   │   ├── musical_components.py      # Support classes
│   │   └── beatbox2drums_model.py     # Beatbox model (separate)
│   ├── data/
│   │   └── melody_dataset.py          # MelodyDataset (CQT-based)
│   └── checkpoints/
│       └── combined_hum2melody_full.pth  # Production model (135MB)
├── frontend/
│   ├── src/
│   │   ├── app/page.tsx        # Main UI
│   │   ├── components/
│   │   │   ├── CodeEditor.tsx  # Monaco integration
│   │   │   ├── MixerPanel.tsx  # Volume controls
│   │   │   └── Timeline/       # Piano-roll editor
│   │   └── lib/
│   │       └── api.ts          # Backend client
│   └── package.json
├── runner/
│   ├── server.js               # Express server
│   ├── MusicCompiler.js        # DSL → Tone.js
│   └── SampleCache.js          # CDN sample management
├── hum2melody_package/         # Standalone ML package
│   ├── models/                 # Model variants
│   ├── scripts/                # Training scripts
│   └── evaluation/             # Evaluation tools
└── scripts/                    # Data labeling utilities
```

## CDN Configuration

Audio samples are hosted on Cloudflare R2:
- Base URL: `https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev`
- Each instrument has `mapping.json` with velocity layers
- Sample paths follow pattern: `instrument/category/name`

## Common Pitfalls

1. **Virtual environment**: Always activate backend venv before running Python commands
2. **Service order**: Start runner before backend (backend needs runner for DSL compilation)
3. **CORS**: Backend must have `ALLOWED_ORIGINS=http://localhost:3000` set
4. **Model paths**: Checkpoint paths are relative to project root, not backend/
5. **Audio bytes**: When calling model inference, include `audio_bytes` field for best results
6. **Git LFS**: Model checkpoints use Git LFS - ensure it's installed for model files

## Git LFS

Large files (models, audio samples) are tracked with Git LFS:
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.wav"
```

Recent commits mention Git LFS setup for model files, so be aware that pulling model checkpoints requires Git LFS.