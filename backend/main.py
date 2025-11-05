from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional

from .settings import load_settings
from .schemas import RunBody, RunnerEvalResponse, IR
from .runner_client import RunnerClient
from . import compiler_stub
from .audio_processor import AudioProcessor
from .model_server import ModelServer
from .database import TrainingDataDB
from .audio_session_manager import get_session_manager
from .hum2melody_endpoints_v2 import (
    hum_to_melody_v2,
    get_segments,
    reprocess_segments,
    delete_session
)

USING_DOCKER=True

settings = load_settings()

# Initialize components
app = FastAPI(title="Music Backend", version="0.2.0")
audio_processor = AudioProcessor(target_sr=16000)
model_server = ModelServer()
db = TrainingDataDB(db_path="training_data.db" if USING_DOCKER else "backend/training_data.db")
session_manager = get_session_manager()  # Audio session manager for re-processing

# Create audio storage directory
AUDIO_STORAGE = Path("audio_uploads" if USING_DOCKER else "backend/audio_uploads")
AUDIO_STORAGE.mkdir(exist_ok=True, parents=True)

# Add to .gitignore
GITIGNORE_PATH = Path(".gitignore" if USING_DOCKER else  "backend/.gitignore")
if not GITIGNORE_PATH.exists():
    GITIGNORE_PATH.write_text("audio_uploads/\n*.db\n")
else:
    content = GITIGNORE_PATH.read_text()
    if "audio_uploads/" not in content:
        with GITIGNORE_PATH.open("a") as f:
            f.write("\naudio_uploads/\n*.db\n")

print("=" * 50)
print("BACKEND STARTUP DEBUG")
print(f"RUNNER_INGEST_URL: {settings.runner_ingest_url}")
print(f"RUNNER_INBOX_PATH: {settings.runner_inbox_path}")
print(f"ALLOWED_ORIGINS: {settings.allowed_origins}")
print(f"Runner configured: {bool(settings.runner_ingest_url or settings.runner_inbox_path)}")
print(f"Audio storage: {AUDIO_STORAGE.absolute()}")
print(f"Database: backend/training_data.db")
print("=" * 50)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

runner = RunnerClient(
    ingest_url=settings.runner_ingest_url,
    inbox_path=settings.runner_inbox_path,
    timeout_s=settings.request_timeout_s
)


def require_ir(ir: IR | None) -> IR:
    if ir is None:
        raise HTTPException(status_code=400, detail="Missing 'ir'")
    return ir


async def save_audio_file(
    upload_file: UploadFile,
    model_type: str
) -> tuple[str, dict]:
    """
    Save uploaded audio file and return path + metadata.
    
    Returns:
        Tuple of (file_path, metadata_dict)
    """
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = upload_file.filename or "audio.wav"
    file_extension = Path(original_filename).suffix or ".wav"
    filename = f"{model_type}_{timestamp}{file_extension}"
    file_path = AUDIO_STORAGE / filename
    
    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Get file metadata
    file_size = file_path.stat().st_size
    
    metadata = {
        "original_filename": original_filename,
        "content_type": upload_file.content_type,
        "file_size_bytes": file_size,
        "upload_timestamp": timestamp
    }
    
    return str(file_path), metadata


@app.get("/health")
def health():
    return {"ok": True, "version": app.version}


@app.get("/stats")
def get_stats():
    """Get training data statistics"""
    try:
        stats = db.get_statistics()
        return {"ok": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/test", response_class=PlainTextResponse)
def test():
    print("TEST ENDPOINT CALLED")
    return compiler_stub.compile_scale_to_dsl()


@app.post("/run", response_model=RunnerEvalResponse)
def run(body: RunBody):
    print("RUN ENDPOINT CALLED")
    print(f"Request body: {body}")
    print(f"Code provided: {body.code is not None}")
    print(f"IR provided: {body.ir is not None}")

    provided = sum(1 for v in [body.code, body.ir] if v is not None)
    if provided != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of 'code' or 'ir'.")

    runner_configured = bool(settings.runner_ingest_url or settings.runner_inbox_path)
    print(f"Runner configured: {runner_configured}")

    try:
        if runner_configured:
            print("FORWARDING TO RUNNER")

            if body.code is not None:
                print("Processing DSL code")
                payload = {
                    "ir": {
                        "metadata": {"tempo": 120},
                        "tracks": [],
                        "__dsl_passthrough": body.code
                    }
                }
            else:
                print("Processing IR data")
                payload = {"ir": require_ir(body.ir).model_dump()}

            print(f"Payload to runner: {payload}")
            result = runner.eval(payload)
            print(f"Runner response: {result}")
            return result

    except Exception as e:
        print(f"Runner error: {e}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=502, detail=f"Runner error: {e}")

    # Fallback
    print("Using local fallback")
    if body.code:
        return RunnerEvalResponse(dsl=body.code, meta={"source": "echo"})
    else:
        dsl = compiler_stub.json_ir_to_dsl(require_ir(body.ir))
        return RunnerEvalResponse(dsl=dsl, meta={"source": "local-stub"})


"""
Main.py excerpt with EXTREME DEBUGGING for hum2melody endpoint
This is just the hum2melody endpoint - replace only this function in your main.py
"""


@app.post("/hum2melody")
async def hum_to_melody(
        audio: UploadFile = File(...),
        save_training_data: bool = Form(True),
        instrument: str = Form("piano/grand_piano_k"),
        onset_high: float = Form(0.30),
        onset_low: float = Form(0.10),
        offset_high: float = Form(0.30),
        offset_low: float = Form(0.10),
        min_confidence: float = Form(0.25),
        return_visualization: bool = Form(True)
):
    """Enhanced hum2melody with interactive tuning support."""
    return await hum_to_melody_v2(
        model_server, audio, save_training_data, instrument,
        onset_high, onset_low, offset_high, offset_low,
        min_confidence, return_visualization
    )


# Original implementation kept as backup (can be removed later)
@app.post("/hum2melody/legacy")
async def hum_to_melody_legacy(
        audio: UploadFile = File(...),
        save_training_data: bool = Form(True),
        instrument: str = Form("piano/grand_piano_k")
):
    print("[HUM2MELODY] ========================================")
    print("[HUM2MELODY] Endpoint called")
    print(f"[HUM2MELODY]   Filename: {audio.filename}")
    print(f"[HUM2MELODY]   Content-Type: {audio.content_type}")
    print(f"[HUM2MELODY]   Instrument: {instrument}")
    print(f"[HUM2MELODY]   Save training data: {save_training_data}")

    try:
        # Read audio bytes FIRST (before any processing)
        print("[HUM2MELODY] Reading audio bytes...")
        audio_bytes = await audio.read()
        print(f"[HUM2MELODY]   ✅ Read {len(audio_bytes)} bytes")

        # Save the bytes directly
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = audio.filename or "audio.wav"
        file_extension = Path(original_filename).suffix or ".wav"
        filename = f"hum2melody_{timestamp}{file_extension}"
        file_path = AUDIO_STORAGE / filename

        print(f"[HUM2MELODY] Saving audio to: {file_path}")
        with file_path.open("wb") as f:
            f.write(audio_bytes)

        file_size = len(audio_bytes)
        file_metadata = {
            "original_filename": original_filename,
            "content_type": audio.content_type,
            "file_size_bytes": file_size,
            "upload_timestamp": timestamp
        }
        print(f"[HUM2MELODY]   ✅ Saved {file_size} bytes")

        # Process audio (for metadata and fallback)
        print("[HUM2MELODY] Processing audio for features...")
        audio_features = audio_processor.preprocess_for_hum2melody(audio_bytes)
        print(f"[HUM2MELODY]   ✅ Duration: {audio_features['duration']:.2f}s")
        print(f"[HUM2MELODY]   Sample rate: {audio_features['sample_rate']}")

        # ADD the raw bytes to features dict (needed by model)
        audio_features["audio_bytes"] = audio_bytes
        audio_features["audio_path"] = str(file_path)
        audio_features["instrument"] = instrument
        print(f"[HUM2MELODY]   ✅ Added audio_bytes ({len(audio_bytes)} bytes)")
        print(f"[HUM2MELODY]   ✅ Added audio_path: {file_path}")

        # Get model prediction
        print("[HUM2MELODY] Calling model_server.predict_melody()...")
        print(f"[HUM2MELODY]   model_server type: {type(model_server)}")
        print(f"[HUM2MELODY]   model_server.predictor is None: {model_server.predictor is None}")

        melody_track = await model_server.predict_melody(audio_features)

        print(f"[HUM2MELODY]   ✅ Returned track")
        print(f"[HUM2MELODY]   Track has notes: {melody_track.notes is not None}")
        if melody_track.notes:
            print(f"[HUM2MELODY]   Number of notes: {len(melody_track.notes)}")

        # Validate track has notes
        if not melody_track.notes:
            print("[HUM2MELODY] ⚠️ No notes predicted, forcing fallback")
            audio_features.pop("audio_bytes", None)
            audio_features.pop("audio_path", None)
            melody_track = await model_server.predict_melody(audio_features)

        print(f"[HUM2MELODY] ✅ Final note count: {len(melody_track.notes)}")

        # Create IR
        print("[HUM2MELODY] Creating IR...")
        ir = IR(
            metadata={
                "tempo": 120,
                "key": "Am",
                "time_signature": "4/4",
                "duration": audio_features['duration']
            },
            tracks=[melody_track]
        )
        print("[HUM2MELODY]   ✅ IR created")

        # Save to database if requested
        audio_id = None
        if save_training_data:
            print("[HUM2MELODY] Saving to database...")
            audio_id = db.save_audio_sample(
                file_path=str(file_path),
                model_type="hum2melody",
                file_format=Path(original_filename).suffix.lstrip('.') or "wav",
                sample_rate=audio_features['sample_rate'],
                duration=audio_features['duration'],
                metadata=file_metadata
            )
            print(f"[HUM2MELODY]   ✅ Saved with ID: {audio_id}")

            # Save prediction
            db.save_prediction(
                audio_sample_id=audio_id,
                model_type="hum2melody",
                prediction=melody_track.model_dump()
            )
            print("[HUM2MELODY]   ✅ Prediction saved")

        model_used = "trained" if model_server.predictor else "mock"
        print(f"[HUM2MELODY] Model used: {model_used}")
        print("[HUM2MELODY] ========================================")

        return JSONResponse(content={
            "status": "success",
            "ir": ir.model_dump(),
            "audio_id": audio_id,
            "metadata": {
                "duration": audio_features['duration'],
                "num_notes": len(melody_track.notes) if melody_track.notes else 0,
                "file_path": str(file_path),
                "model_used": model_used,
                "instrument": instrument
            }
        })

    except Exception as e:
        print(f"[HUM2MELODY] ❌ ERROR: {e}")
        import traceback
        print("[HUM2MELODY] Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/beatbox2drums")
async def beatbox_to_drums(
    audio: UploadFile = File(...),
    save_training_data: bool = Form(True)
):
    """
    Process beatbox audio and return drum pattern in IR format.
    
    Args:
        audio: Audio file (WAV or MP3)
        save_training_data: Whether to save for future model training
        
    Returns:
        JSON with drum pattern IR and metadata
    """
    print("[BEATBOX2DRUMS] Endpoint called")
    print(f"File: {audio.filename}, Type: {audio.content_type}")
    
    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        
        # Save audio file
        file_path, file_metadata = await save_audio_file(audio, "beatbox2drums")
        print(f"Saved audio to: {file_path}")
        
        # Process audio
        print("Processing audio...")
        audio_features = audio_processor.preprocess_for_beatbox(audio_bytes)
        print(f"Extracted features: duration={audio_features['duration']:.2f}s, tempo={audio_features['tempo']:.1f}")
        
        # Get model prediction
        print("Getting model prediction...")
        drums_track = await model_server.predict_drums(audio_features)
        
        # Create IR
        ir = IR(
            metadata={
                "tempo": int(audio_features['tempo']),
                "time_signature": "4/4",
                "duration": audio_features['duration']
            },
            tracks=[drums_track]
        )
        
        # Save to database if requested
        audio_id = None
        if save_training_data:
            original_filename = audio.filename or "audio.wav"
            audio_id = db.save_audio_sample(
                file_path=file_path,
                model_type="beatbox2drums",
                file_format=Path(original_filename).suffix.lstrip('.') or "wav",
                sample_rate=audio_features['sample_rate'],
                duration=audio_features['duration'],
                metadata=file_metadata
            )
            print(f"Saved to database with ID: {audio_id}")
            
            # Save prediction
            db.save_prediction(
                audio_sample_id=audio_id,
                model_type="beatbox2drums",
                prediction=drums_track.model_dump()
            )
        
        return JSONResponse(content={
            "status": "success",
            "ir": ir.model_dump(),
            "audio_id": audio_id,
            "metadata": {
                "duration": audio_features['duration'],
                "tempo": audio_features['tempo'],
                "num_samples": len(drums_track.samples) if drums_track.samples else 0,
                "file_path": file_path
            }
        })
        
    except Exception as e:
        print(f"Error in beatbox2drums: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/arrange")
async def arrange_track(body: dict):
    """
    Take existing IR and add accompanying tracks (bass, chords, drums).
    
    Request body should contain:
    {
        "ir": <existing IR object>,
        "style": "pop" | "jazz" | "electronic" (optional)
    }
    
    Returns:
        Enhanced IR with additional tracks
    """
    print("[ARRANGE] Endpoint called")
    
    try:
        # Parse input IR
        if "ir" not in body:
            raise HTTPException(status_code=400, detail="Missing 'ir' in request body")
        
        existing_ir = IR(**body["ir"])
        style = body.get("style", "pop")
        
        print(f"Arranging {len(existing_ir.tracks)} existing track(s) in {style} style")
        
        # Get arrangement from model
        enhanced_ir = await model_server.arrange_track(existing_ir, style=style)
        
        print(f"Generated {len(enhanced_ir.tracks)} total tracks")
        
        return JSONResponse(content={
            "status": "success",
            "ir": enhanced_ir.model_dump(),
            "metadata": {
                "original_tracks": len(existing_ir.tracks),
                "total_tracks": len(enhanced_ir.tracks),
                "added_tracks": len(enhanced_ir.tracks) - len(existing_ir.tracks),
                "style": style
            }
        })
        
    except Exception as e:
        print(f"Error in arrange: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error arranging track: {str(e)}")


@app.post("/feedback")
async def submit_feedback(
    audio_id: int = Form(...),
    rating: int = Form(...),
    prediction_id: Optional[int] = Form(None),
    feedback_text: Optional[str] = Form(None)
):
    """
    Submit user feedback for a prediction.
    
    Args:
        audio_id: Database ID of audio sample
        rating: User rating (1-5)
        prediction_id: Optional ID of specific prediction
        feedback_text: Optional text feedback
        
    Returns:
        Success confirmation
    """
    print(f"[FEEDBACK] Received: audio_id={audio_id}, rating={rating}")
    
    try:
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        feedback_id = db.save_feedback(
            audio_sample_id=audio_id,
            rating=rating,
            prediction_id=prediction_id,
            feedback_text=feedback_text
        )
        
        return JSONResponse(content={
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Thank you for your feedback!"
        })
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


# ============================================================
# Interactive Tuning Endpoints
# ============================================================

@app.get("/hum2melody/segments/{session_id}")
async def get_session_segments(session_id: str):
    """Get current segment detection for a session."""
    return await get_segments(session_id)


@app.post("/hum2melody/reprocess")
async def reprocess(
    session_id: str = Form(...),
    manual_onsets: str = Form(...),
    manual_offsets: str = Form(...)
):
    """Reprocess audio with user-provided onset/offset markers."""
    return await reprocess_segments(
        model_server, session_id, manual_onsets, manual_offsets
    )


@app.delete("/hum2melody/session/{session_id}")
async def delete_audio_session(session_id: str):
    """Delete a session and clean up files."""
    return await delete_session(session_id)


# Cleanup task for expired sessions
@app.on_event("startup")
async def start_session_cleanup():
    """Start background task to clean up expired sessions every hour."""
    import asyncio
    async def cleanup_loop():
        while True:
            await asyncio.sleep(3600)  # 1 hour
            count = session_manager.cleanup_expired_sessions()
            if count > 0:
                print(f"[SessionCleanup] Removed {count} expired sessions")
    asyncio.create_task(cleanup_loop())