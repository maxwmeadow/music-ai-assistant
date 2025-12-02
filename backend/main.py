from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional

from .settings import load_settings
from .schemas import RunBody, RunnerEvalResponse, IR, GenerateTrackRequest
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
from .job_manager import get_job_manager, JobStatus
from .arranger_service import get_arranger_service
import asyncio

USING_DOCKER=True

settings = load_settings()

# Initialize components
app = FastAPI(title="Music Backend", version="0.2.0")

@app.on_event("startup")
async def startup_event():
    """Download models from R2 on startup (for Railway deployment)"""
    print("Checking for model checkpoints...")
    try:
        from .download_models import check_and_download_models
        check_and_download_models()
    except Exception as e:
        print(f"⚠️ Model download check failed: {e}")
        print("Will attempt to use local models or fall back to mock predictions")

audio_processor = AudioProcessor(target_sr=16000)
model_server = ModelServer()
db = TrainingDataDB(db_path="training_data.db" if USING_DOCKER else "backend/training_data.db")
session_manager = get_session_manager()  # Audio session manager for re-processing
job_manager = get_job_manager()  # Job manager for async processing

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
    allow_credentials=True,
    expose_headers=["*"]
)

# Global exception handler to ensure CORS headers even on errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return JSON with CORS headers"""
    import traceback
    error_detail = str(exc)
    error_traceback = traceback.format_exc()

    print(f"[GLOBAL EXCEPTION HANDLER] Caught exception: {error_detail}")
    print(f"[GLOBAL EXCEPTION HANDLER] Traceback:\n{error_traceback}")

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": error_detail,
            "type": type(exc).__name__
        },
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true"
        }
    )

runner = RunnerClient(
    ingest_url=settings.runner_ingest_url,
    inbox_path=settings.runner_inbox_path,
    timeout_s=settings.request_timeout_s
)


# Helper function to offset notes in IR by a given start time
def offset_ir_notes(ir: dict, start_time: float) -> dict:
    """
    Offset all note start times in an IR by the given start_time.
    This allows placing generated notes at a specific point in the song.

    Args:
        ir: The IR dictionary containing tracks with notes
        start_time: Time offset in seconds to add to all note start times

    Returns:
        Modified IR with offset notes
    """
    if start_time == 0.0:
        return ir  # No offset needed

    if "tracks" in ir:
        for track in ir["tracks"]:
            if "notes" in track and track["notes"]:
                for note in track["notes"]:
                    note["start"] += start_time
            if "samples" in track and track["samples"]:
                for sample in track["samples"]:
                    sample["start"] += start_time

    return ir


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
    """Health check with model loading status"""
    model_status = {
        "loaded": model_server.predictor is not None,
        "loading_attempted": model_server._model_loading_attempted,
        "checkpoint_exists": model_server.checkpoint_path.exists()
    }
    return {
        "ok": True,
        "version": app.version,
        "model": model_status
    }


@app.get("/stats")
def get_stats():
    """Get training data statistics"""
    try:
        stats = db.get_statistics()
        return {"ok": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Poll job status for async processing.

    Returns:
        - status: pending, processing, completed, or failed
        - result: If completed, contains the full response
        - error: If failed, contains error message
        - progress: 0-100 progress indicator
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse(content=job.to_dict())


@app.get("/jobs")
def list_jobs():
    """Get statistics about all jobs"""
    return JSONResponse(content=job_manager.get_stats())


@app.get("/test", response_class=PlainTextResponse)
def test():
    print("TEST ENDPOINT CALLED")
    return compiler_stub.compile_scale_to_dsl()


@app.post("/run", response_model=RunnerEvalResponse)
def run(body: RunBody):
    print(f"[RUN] code={'yes' if body.code else 'no'}, ir={'yes' if body.ir else 'no'}")

    # Allow both code and IR to be provided together
    provided = sum(1 for v in [body.code, body.ir] if v is not None)
    if provided == 0:
        raise HTTPException(status_code=400, detail="Provide at least one of 'code' or 'ir'.")

    runner_configured = bool(settings.runner_ingest_url or settings.runner_inbox_path)

    try:
        if runner_configured:
            # If both code and IR are provided, merge them
            if body.code is not None and body.ir is not None:
                ir_data = require_ir(body.ir).model_dump()
                ir_data["__dsl_passthrough"] = body.code
                payload = {"ir": ir_data}
            elif body.code is not None:
                payload = {
                    "ir": {
                        "metadata": {"tempo": 120},
                        "tracks": [],
                        "__dsl_passthrough": body.code
                    }
                }
            else:
                payload = {"ir": require_ir(body.ir).model_dump()}

            result = runner.eval(payload)
            return result

    except Exception as e:
        print(f"[RUN] ❌ Runner error: {e}")
        raise HTTPException(status_code=502, detail=f"Runner error: {e}")

    # Fallback
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
        instrument: str = Form("piano/steinway_grand"),
        onset_high: float = Form(0.30),
        onset_low: float = Form(0.10),
        offset_high: float = Form(0.30),
        offset_low: float = Form(0.10),
        min_confidence: float = Form(0.25),
        return_visualization: bool = Form(False),  # Default False for production performance
        async_mode: bool = Form(False),  # Enable async processing for long requests
        start_time: float = Form(0.0)  # Time offset in song for placing notes
):
    """
    Enhanced hum2melody with interactive tuning support.

    Args:
        async_mode: If True, returns job_id immediately and processes in background.
                   Use /jobs/{job_id} to poll for results.
    """
    if async_mode:
        # Read audio bytes immediately (before background task)
        audio_bytes = await audio.read()
        original_filename = audio.filename or "recording.wav"
        content_type = audio.content_type

        # Create job and process in background
        job_id = job_manager.create_job()

        # Start background task
        async def process_in_background():
            try:
                job_manager.update_status(job_id, JobStatus.PROCESSING, progress=10)

                # Save audio to temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_extension = Path(original_filename).suffix or ".wav"
                filename = f"hum2melody_{timestamp}_{job_id[:8]}{file_extension}"
                file_path = AUDIO_STORAGE / filename

                print(f"[ASYNC HUM2MELODY] Saving audio for job {job_id}: {file_path}")
                with file_path.open("wb") as f:
                    f.write(audio_bytes)

                # Create a mock UploadFile for processing
                from io import BytesIO
                from fastapi import UploadFile

                audio_file = UploadFile(
                    file=BytesIO(audio_bytes),
                    filename=original_filename,
                    headers={"content-type": content_type}
                )

                # Process the audio
                result = await hum_to_melody_v2(
                    model_server, audio_file, save_training_data, instrument,
                    onset_high, onset_low, offset_high, offset_low,
                    min_confidence, return_visualization
                )

                # Extract result content
                if hasattr(result, 'body'):
                    import json
                    result_data = json.loads(result.body.decode())
                else:
                    result_data = result

                # Apply start_time offset if provided
                if start_time > 0.0 and 'ir' in result_data:
                    result_data['ir'] = offset_ir_notes(result_data['ir'], start_time)

                # Store result
                job_manager.set_result(job_id, result_data)

            except Exception as e:
                print(f"[ASYNC HUM2MELODY] Job {job_id} failed: {e}")
                import traceback
                traceback.print_exc()
                job_manager.set_error(job_id, str(e))

        # Schedule background task
        asyncio.create_task(process_in_background())

        return JSONResponse(content={
            "status": "accepted",
            "job_id": job_id,
            "message": "Processing in background. Poll /jobs/{job_id} for results."
        }, status_code=202)
    else:
        # Synchronous processing (original behavior)
        result = await hum_to_melody_v2(
            model_server, audio, save_training_data, instrument,
            onset_high, onset_low, offset_high, offset_low,
            min_confidence, return_visualization
        )

        # Apply start_time offset if provided
        if start_time > 0.0:
            # Extract result content
            if hasattr(result, 'body'):
                import json
                result_data = json.loads(result.body.decode())
            else:
                result_data = result

            # Apply offset and return
            if 'ir' in result_data:
                result_data['ir'] = offset_ir_notes(result_data['ir'], start_time)

            return JSONResponse(content=result_data)

        return result


# Original implementation kept as backup (can be removed later)
@app.post("/hum2melody/legacy")
async def hum_to_melody_legacy(
        audio: UploadFile = File(...),
        save_training_data: bool = Form(True),
        instrument: str = Form("piano/steinway_grand")
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
    save_training_data: bool = Form(True),
    return_visualization: bool = Form(False),
    start_time: float = Form(0.0)  # Time offset in song for placing drum hits
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

        # Add audio_bytes and audio_path for model inference
        audio_features['audio_bytes'] = audio_bytes
        audio_features['audio_path'] = str(file_path)

        # Get model prediction
        print("Getting model prediction...")
        drums_track, visualization_data = await model_server.predict_drums(audio_features, return_visualization=return_visualization)
        
        # Create IR
        ir = IR(
            metadata={
                "tempo": int(audio_features['tempo']),
                "time_signature": "4/4",
                "duration": audio_features['duration']
            },
            tracks=[drums_track]
        )

        # Apply start_time offset if provided
        if start_time > 0.0:
            ir_dict = ir.model_dump()
            ir_dict = offset_ir_notes(ir_dict, start_time)
            ir = IR(**ir_dict)

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
        
        response_data = {
            "status": "success",
            "ir": ir.model_dump(),
            "audio_id": audio_id,
            "metadata": {
                "duration": audio_features['duration'],
                "tempo": audio_features['tempo'],
                "num_samples": len(drums_track.samples) if drums_track.samples else 0,
                "file_path": file_path
            }
        }

        # Add visualization data if requested
        if return_visualization and visualization_data:
            response_data["visualization"] = visualization_data

        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error in beatbox2drums: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/arrange")
async def arrange_track(body: GenerateTrackRequest):
    """
    Generate a complementary track using LLM-based arranger.

    Request body should contain:
    {
        "dsl_code": <current DSL code>,
        "track_type": "bass" | "chords" | "pad" | "melody" | "counterMelody" | "arpeggio" | "drums",
        "genre": "pop" | "jazz" | "electronic" | etc.,
        "custom_request": "optional additional instructions"
    }

    Returns:
        Generated DSL code for the new track
    """
    print("[ARRANGE] LLM-based track generation endpoint called")

    try:
        print(f"Generating {body.track_type} track in {body.genre} style")
        if body.custom_request:
            print(f"Custom request: {body.custom_request}")

        # Get arranger service
        arranger = get_arranger_service()

        # Generate the track
        generated_dsl = arranger.generate_arrangement(
            current_dsl=body.dsl_code,
            track_type=body.track_type,
            genre=body.genre,
            custom_request=body.custom_request,
            creativity=body.creativity,
            complexity=body.complexity
        )

        print(f"Successfully generated {body.track_type} track")

        return JSONResponse(content={
            "status": "success",
            "generated_dsl": generated_dsl,
            "metadata": {
                "track_type": body.track_type,
                "genre": body.genre,
                "has_custom_request": body.custom_request is not None
            }
        })

    except Exception as e:
        print(f"Error in arrange: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating track: {str(e)}")


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


# Cleanup task for expired sessions and jobs
@app.on_event("startup")
async def start_cleanup_tasks():
    """Start background tasks to clean up expired sessions and jobs every hour."""
    async def cleanup_loop():
        while True:
            await asyncio.sleep(3600)  # 1 hour

            # Clean up expired sessions
            session_count = session_manager.cleanup_expired_sessions()
            if session_count > 0:
                print(f"[SessionCleanup] Removed {session_count} expired sessions")

            # Clean up expired jobs
            job_count = job_manager.cleanup_expired(max_age_hours=1)
            if job_count > 0:
                print(f"[JobCleanup] Removed {job_count} expired jobs")

    asyncio.create_task(cleanup_loop())