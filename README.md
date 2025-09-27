# Music AI Assistant

A web-based music generation and playback system with AI assistance. Generate musical compositions through natural language, edit them in a Monaco code editor, and play them back using Tone.js audio synthesis.

## Architecture

- **Frontend**: Next.js with Monaco Editor for code editing
- **Backend**: FastAPI Python server handling API requests
- **Runner**: Node.js service that processes music code and generates Tone.js audio
- **Audio**: Browser-based playback using Tone.js Web Audio API

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Git**

## Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd music-ai-assistant
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Go back to project root
cd ..
```

### 3. Runner Setup
```bash
cd runner
npm install
cd ..
```

### 4. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 5. Environment Configuration

Create environment files for configuration:

**Backend** - Create `backend/.env`:
```
ALLOWED_ORIGINS=http://localhost:3000
RUNNER_INGEST_URL=http://localhost:5001/eval
RUNNER_INBOX_PATH=
REQUEST_TIMEOUT_S=3
```

**Frontend** - Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Option 1: Automated Startup (Windows)
Use the provided PowerShell script to start all services:
```powershell
.\start-all.ps1
```

This will open three terminal windows:
- Runner (port 5001)
- Backend (port 8000) 
- Frontend (port 3000)

### Option 2: Manual Startup
Open three separate terminals and run each service:

**Terminal 1 - Runner:**
```bash
cd runner
npm run dev
```

**Terminal 2 - Backend:**
```bash
# Set environment variable and activate venv
$env:RUNNER_INGEST_URL="http://localhost:5001/eval"
.\.venv\Scripts\activate
uvicorn backend.main:app --reload
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm run dev
```

## Using the Application

1. Open your browser to `http://localhost:3000`
2. Click "Generate (/test)" to load sample music code
3. Edit the code in the Monaco editor if desired
4. Click "Send to Runner" to process the music
5. Click "Play" to hear the generated audio
6. Use "Stop" to halt playback

## API Endpoints

- `GET /health` - Backend health check
- `GET /test` - Generate sample music code
- `POST /run` - Process music code and return executable audio
- `GET localhost:5001/health` - Runner health check

## Troubleshooting

### No Audio Playback
- Ensure all three services are running
- Check browser console for JavaScript errors
- Verify Tone.js is loaded (check Network tab in DevTools)
- Make sure browser allows audio (some browsers require user interaction)

### Backend Connection Issues
- Verify `RUNNER_INGEST_URL` environment variable is set
- Check that Runner is responding at `http://localhost:5001/health`
- Ensure CORS is properly configured for frontend origin

### Port Conflicts
If ports 3000, 8000, or 5001 are in use:
1. Stop conflicting processes
2. Or modify port numbers in configuration files
3. Update environment variables accordingly

### Process Cleanup
To stop all services on Windows:
```powershell
.\stop-all.ps1
```

Or manually find and kill processes:
```powershell
# Find processes using ports
netstat -ano | findstr ":3000"
netstat -ano | findstr ":8000" 
netstat -ano | findstr ":5001"

# Kill by PID (replace XXXX with actual PID)
taskkill /PID XXXX /F
```

## Development Notes

- The system uses a Domain Specific Language (DSL) for music representation
- Music code is processed through: Frontend → Backend → Runner → Tone.js
- Hot reload is enabled for all services during development
- The runner converts musical concepts into executable JavaScript audio code

## File Structure
```
music-ai-assistant/
├── backend/           # FastAPI Python server
├── runner/           # Node.js music processing service  
├── frontend/         # Next.js React application
├── start-all.ps1    # Windows startup script
└── README.md        # This file
```