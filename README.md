# Music AI Assistant

A web-based music generation and playback system with AI assistance. Generate musical compositions through natural language, edit them in a Monaco code editor, and play them back using Tone.js audio synthesis with a professional mixing interface and timeline editor.

## Architecture

- **Frontend**: Next.js with Monaco Editor, real-time mixer, and interactive timeline
- **Backend**: FastAPI Python server handling API requests and DSL compilation
- **Runner**: Node.js service that compiles music DSL to executable Tone.js code
- **Audio**: Browser-based playback using Tone.js with persistent sample caching

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **npm** or **yarn**
- **Git**

## Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd music-ai-assistant
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

**Backend Dependencies** (in `requirements.txt`):
```
fastapi
uvicorn[standard]
pydantic
python-multipart
httpx
```

### 3. Runner Setup

```bash
cd runner

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

**Runner Dependencies** (in `package.json`):
```json
{
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  }
}
```

The runner uses the following modules (should be present in `runner/` directory):
- `server.js` - Main Express server with DSL compilation
- `parser.js` - Music JSON parser
- `SampleCache.js` - CDN sample management
- `InstrumentFactory.js` - Instrument creation and pooling
- `MusicScheduler.js` - Note scheduling system
- `MusicCompiler.js` - DSL to Tone.js compilation

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Return to project root
cd ..
```

**Frontend Dependencies** (in `package.json`):
```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@monaco-editor/react": "^4.6.0",
    "lucide-react": "^0.300.0",
    "tone": "^14.7.77"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "typescript": "^5",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32"
  }
}
```

### 5. Environment Configuration

**Backend** - Create `backend/.env`:
```env
ALLOWED_ORIGINS=http://localhost:3000
RUNNER_INGEST_URL=http://localhost:5001/eval
RUNNER_INBOX_PATH=
REQUEST_TIMEOUT_S=30
```

**Frontend** - Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Option 1: Automated Startup (Windows PowerShell)

```powershell
.\start-all.ps1
```

This opens three terminal windows:
- **Runner** (port 5001) - DSL compilation service
- **Backend** (port 8000) - API server
- **Frontend** (port 3000) - Web interface

### Option 2: Manual Startup

Open three separate terminals:

**Terminal 1 - Runner:**
```bash
cd runner
npm start
```

**Terminal 2 - Backend:**
```bash
# Activate virtual environment first
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Set environment variables (Windows PowerShell):
$env:RUNNER_INGEST_URL="http://localhost:5001/eval"
$env:ALLOWED_ORIGINS="http://localhost:3000"

# Start backend
uvicorn backend.main:app --reload
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm run dev
```

## Using the Application

1. Open `http://localhost:3000` in your browser
2. Click **"Load Sample Code"** to load example music DSL
3. Click **"Compile"** to process the code through the runner
4. Adjust track volumes in the **Mixer Panel**
5. Click **"Play"** to hear the generated audio
6. Use the **Timeline Editor** to visually edit notes:
   - Drag notes horizontally to change timing
   - Drag the right edge to change duration
   - Click to select, press Delete/Backspace to remove
7. Click **"Stop"** to halt playback

## Key Features

### Mixer Panel
- Per-track volume faders (-40dB to +12dB)
- Mute (M) and Solo (S) buttons
- Real-time volume adjustment during playback
- Visual instrument labels

### Timeline Editor
- Visual piano-roll style note editing
- Drag notes to change timing
- Resize notes to change duration
- Delete notes with keyboard shortcuts
- Zoom control for detailed editing
- Playback position indicator
- Chord visualization (blue blocks)

### Audio System
- **Persistent sample caching** - Samples remain loaded between plays
- **Polyphonic voice pools** - 8 voices per instrument for chord support
- **Velocity layer selection** - Intelligent sample selection based on instrument type
- **Drum mapping** - Automatic note-to-sample mapping for drum kits

## API Endpoints

- `GET /health` - Backend health check
- `GET /test` - Generate sample DSL code
- `POST /run` - Compile DSL code to executable Tone.js
- `GET localhost:5001/health` - Runner health check
- `POST localhost:5001/eval` - DSL/IR compilation endpoint

## DSL Format Example

```javascript
tempo(128)

track("melody") {
  instrument("piano/grand_piano_k")
  note("E4", 0.5, 0.8)
  note("D4", 0.5, 0.7)
  note("C4", 0.5, 0.8)
}

track("chords") {
  instrument("synth/pad/pd_fatness_pad")
  chord(["C4", "E4", "G4"], 2.0, 0.6)
  chord(["F4", "A4", "C5"], 2.0, 0.6)
}

track("drums") {
  instrument("drums/bedroom_drums")
  note("C2", 0.5, 1.0)   // Kick
  note("F#2", 0.5, 0.7)  // Hi-hat
  note("D2", 0.5, 1.0)   // Snare
}
```

## Troubleshooting

### No Audio Playback
- Verify all three services are running (check terminal outputs)
- Check browser console for errors (F12 → Console)
- Ensure you clicked "Compile" before "Play"
- Check that browser allows audio (some browsers require user interaction first)

### Backend Connection Issues
- Verify `RUNNER_INGEST_URL` environment variable is set correctly
- Test runner health: `curl http://localhost:5001/health`
- Check CORS settings in backend `.env` file

### Compilation Errors
- Check runner console for detailed error messages
- Verify DSL syntax matches the format shown above
- Ensure instrument paths exist in the CDN catalog

### Port Conflicts

If ports 3000, 8000, or 5001 are already in use:

**Windows:**
```powershell
# Find processes
netstat -ano | findstr ":3000"
netstat -ano | findstr ":8000"
netstat -ano | findstr ":5001"

# Kill by PID
taskkill /PID <PID> /F
```

**Mac/Linux:**
```bash
# Find and kill processes
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
lsof -ti:5001 | xargs kill -9
```

### Stopping All Services

**Windows:**
```powershell
.\stop-all.ps1
```

**Mac/Linux:**
```bash
# Kill by process name
pkill -f "node.*runner"
pkill -f "uvicorn"
pkill -f "next-server"
```

## Development Notes

- Hot reload is enabled for all services
- Sample metadata is cached in `window.__musicCache` for performance
- The runner compiles DSL to standalone executable Tone.js code
- Timeline changes are reflected back into the DSL code in real-time
- Instrument pools persist between playbacks to avoid reload delays

## CDN Configuration

The system uses Cloudflare R2 for sample storage:
- Base URL: `https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev`
- Samples are organized by instrument type
- Each instrument has a `mapping.json` with velocity layers

## File Structure

```
music-ai-assistant/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── compiler_stub.py     # IR to DSL conversion
│   ├── schemas.py           # Pydantic models
│   └── requirements.txt     # Python dependencies
├── runner/
│   ├── server.js            # Express server with compilation
│   ├── SampleCache.js       # Sample metadata management
│   ├── InstrumentFactory.js # Instrument creation
│   ├── MusicScheduler.js    # Event scheduling
│   ├── MusicCompiler.js     # DSL compilation
│   └── package.json         # Node dependencies
├── frontend/
│   ├── app/
│   │   └── page.tsx         # Main UI component
│   ├── components/
│   │   ├── CodeEditor.tsx   # Monaco editor wrapper
│   │   ├── MixerPanel.tsx   # Audio mixer interface
│   │   └── Timeline/
│   │       └── Timeline.tsx # Visual note editor
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   └── dslParser.ts     # DSL parsing utilities
│   └── package.json         # Frontend dependencies
├── start-all.ps1            # Windows startup script
├── stop-all.ps1             # Generated stop script
└── README.md                # This file
```