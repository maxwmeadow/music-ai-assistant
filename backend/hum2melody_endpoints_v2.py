"""Enhanced Hum2Melody Endpoints with Interactive Tuning"""

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import time

from backend.audio_session_manager import get_session_manager
from backend.segment_extractor import extract_segments_with_detection
from backend.schemas import IR, Track, Note
from backend.database import TrainingDataDB
from backend.audio_processor import AudioProcessor
from backend.model_server import ModelServer


# Initialize dependencies
session_manager = get_session_manager()
db = TrainingDataDB()
audio_processor = AudioProcessor()


async def hum_to_melody_v2(
    model_server: ModelServer,
    audio: UploadFile = File(...),
    save_training_data: bool = Form(True),
    instrument: str = Form("piano/steinway_grand"),
    onset_high: float = Form(0.30),
    onset_low: float = Form(0.10),
    offset_high: float = Form(0.30),
    offset_low: float = Form(0.10),
    min_confidence: float = Form(0.25),
    return_visualization: bool = Form(True)
):
    """Enhanced hum2melody endpoint with session management and visualization data."""
    start_time = time.time()
    print(f"[HUM2MELODY_V2] {audio.filename} | onset={onset_high}/{onset_low} offset={offset_high}/{offset_low} conf={min_confidence}")

    try:
        # Read audio bytes
        t0 = time.time()
        audio_bytes = await audio.read()
        print(f"  ⏱️ Audio read: {time.time() - t0:.2f}s")

        # DEBUG: Log audio info
        print(f"  Audio bytes received: {len(audio_bytes)} bytes")
        print(f"  Audio filename: {audio.filename}")
        print(f"  Audio content_type: {audio.content_type}")
        print(f"  First 44 bytes (WAV header): {audio_bytes[:44].hex() if len(audio_bytes) >= 44 else 'too short'}")

        # Create session
        t1 = time.time()
        session_metadata = {
            'instrument': instrument,
            'onset_high': onset_high,
            'onset_low': onset_low,
            'offset_high': offset_high,
            'offset_low': offset_low,
            'min_confidence': min_confidence
        }

        session_id = session_manager.create_session(
            audio_bytes,
            audio.filename or "recording.wav",
            metadata=session_metadata
        )
        print(f"  ⏱️ Session created: {time.time() - t1:.2f}s")

        # Get audio path
        audio_path = session_manager.get_audio_path(session_id)

        # Process audio for features (get duration)
        t2 = time.time()
        audio_features = audio_processor.preprocess_for_hum2melody(audio_bytes)
        duration = audio_features['duration']
        print(f"  ⏱️ Audio preprocessing: {time.time() - t2:.2f}s")

        # Use ONE detection system: amplitude onset detection + pitch model
        # This is the same detection used in the DetectionTuner visualization
        print(f"  Running detection with amplitude onset detector...")
        t3 = time.time()
        viz_data = extract_segments_with_detection(
            audio_path,
            onset_high=onset_high,
            onset_low=onset_low,
            offset_high=offset_high,
            offset_low=offset_low,
            min_confidence=min_confidence
        )
        print(f"  ⏱️ SEGMENT EXTRACTION (model inference): {time.time() - t3:.2f}s")

        print(f"  Detected {len(viz_data['segments'])} segments")

        # Create track from segments
        notes = []
        for seg in viz_data['segments']:
            notes.append(Note(
                pitch=seg['pitch'],
                start=seg['start'],
                duration=seg['duration'],
                velocity=seg['confidence']  # Use confidence as velocity
            ))

        melody_track = Track(
            id='melody',
            instrument=instrument,
            notes=notes
        )

        print(f"  ✅ Created track with {len(notes)} notes")

        # Create IR
        ir = IR(
            metadata={
                "tempo": 120,
                "key": "Am",
                "time_signature": "4/4",
                "duration": duration
            },
            tracks=[melody_track]
        )

        # Save to database
        audio_id = None
        if save_training_data:
            audio_id = db.save_audio_sample(
                file_path=audio_path,
                model_type="hum2melody",
                file_format=Path(audio.filename or "recording.wav").suffix.lstrip('.') or "wav",
                sample_rate=16000,
                duration=duration,
                metadata={'session_id': session_id, **session_metadata}
            )
            db.save_prediction(
                audio_sample_id=audio_id,
                model_type="hum2melody",
                prediction=melody_track.model_dump()
            )

        # Get waveform data for visualization
        waveform_data = session_manager.get_waveform_data(session_id, max_samples=2000)

        # Build response (always include visualization since we already computed it)
        response_data = {
            "status": "success",
            "ir": ir.model_dump(),
            "session_id": session_id,
            "audio_id": audio_id,
            "visualization": {
                "segments": viz_data['segments'],
                "onsets": viz_data['onsets'],
                "offsets": viz_data['offsets'],
                "waveform": waveform_data,
                "parameters": viz_data['parameters']
            },
            "metadata": {
                "duration": duration,
                "num_notes": len(notes),
                "instrument": instrument,
                "parameters": {
                    "onset_high": onset_high,
                    "onset_low": onset_low,
                    "offset_high": offset_high,
                    "offset_low": offset_low,
                    "min_confidence": min_confidence
                }
            }
        }

        total_time = time.time() - start_time
        print(f"  ⏱️ TOTAL REQUEST TIME: {total_time:.2f}s")

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"[HUM2MELODY_V2] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


async def get_segments(session_id: str):
    """Get current segment detection for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    audio_path = session['audio_path']
    metadata = session.get('metadata', {})

    # Extract with stored parameters
    viz_data = extract_segments_with_detection(
        audio_path,
        onset_high=metadata.get('onset_high', 0.30),
        onset_low=metadata.get('onset_low', 0.10),
        offset_high=metadata.get('offset_high', 0.30),
        offset_low=metadata.get('offset_low', 0.10),
        min_confidence=metadata.get('min_confidence', 0.25)
    )

    # Get waveform
    waveform_data = session_manager.get_waveform_data(session_id)

    return JSONResponse(content={
        "status": "success",
        "session_id": session_id,
        "visualization": {
            "segments": viz_data['segments'],
            "onsets": viz_data['onsets'],
            "offsets": viz_data['offsets'],
            "waveform": waveform_data,
            "parameters": viz_data['parameters']
        }
    })


async def reprocess_segments(
    model_server: ModelServer,
    session_id: str = Form(...),
    manual_onsets: str = Form(...),
    manual_offsets: str = Form(...)
):
    """Reprocess audio with user-provided onset/offset markers."""
    print(f"[REPROCESS] Session: {session_id}")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    audio_path = session['audio_path']
    instrument = session['metadata'].get('instrument', 'piano/steinway_grand')

    try:
        # Parse markers
        import json
        onset_list = json.loads(manual_onsets)
        offset_list = json.loads(manual_offsets)

        print(f"  Reprocessing with {len(onset_list)} onsets, {len(offset_list)} offsets")

        if len(onset_list) == 0 or len(offset_list) == 0:
            raise HTTPException(status_code=400, detail="Must provide at least one onset and one offset")

        # Extract segments with user's markers
        viz_data = extract_segments_with_detection(
            audio_path,
            manual_onsets=onset_list,
            manual_offsets=offset_list
        )

        # Convert segments to Notes
        notes = []
        for seg in viz_data['segments']:
            notes.append(Note(
                pitch=seg['pitch'],
                start=seg['start'],
                duration=seg['duration'],
                velocity=int(seg['confidence'] * 127)
            ))

        print(f"  Generated {len(notes)} notes from user markers")

        # Create track
        melody_track = Track(
            id='melody',
            instrument=instrument,
            notes=notes
        )

        # Get duration
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr

        # Create IR
        ir = IR(
            metadata={
                "tempo": 120,
                "key": "Am",
                "time_signature": "4/4",
                "duration": float(duration)
            },
            tracks=[melody_track]
        )

        # Get waveform
        waveform_data = session_manager.get_waveform_data(session_id)

        return JSONResponse(content={
            "status": "success",
            "ir": ir.model_dump(),
            "session_id": session_id,
            "visualization": {
                "segments": viz_data['segments'],
                "onsets": viz_data['onsets'],
                "offsets": viz_data['offsets'],
                "waveform": waveform_data,
                "parameters": viz_data['parameters']
            },
            "metadata": {
                "num_notes": len(notes),
                "duration": duration
            }
        })

    except Exception as e:
        print(f"[REPROCESS] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reprocessing: {str(e)}")


async def delete_session(session_id: str):
    """Delete a session and clean up files."""
    success = session_manager.delete_session(session_id)

    if success:
        return JSONResponse(content={"status": "success", "message": "Session deleted"})
    else:
        raise HTTPException(status_code=404, detail="Session not found")
