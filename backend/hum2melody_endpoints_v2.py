"""
Enhanced Hum2Melody Endpoints with Interactive Tuning

New endpoints for interactive detection tuning:
- POST /hum2melody - Enhanced to return session_id + visualization data
- GET /hum2melody/segments/{session_id} - Get current segments
- POST /hum2melody/reprocess - Reprocess with new parameters or manual segments
- DELETE /hum2melody/session/{session_id} - Clean up session

This should REPLACE the existing /hum2melody endpoint in main.py
"""

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from backend.audio_session_manager import get_session_manager
from backend.segment_extractor import extract_segments_with_detection
from backend.schemas import IR, Track, Note
from backend.database import TrainingDataDB
from backend.audio_processor import AudioProcessor
from backend.model_server import ModelServer


# Initialize dependencies (these should be passed in or imported properly)
session_manager = get_session_manager()
db = TrainingDataDB()
audio_processor = AudioProcessor()


# ============================================================
# Enhanced /hum2melody endpoint
# ============================================================

async def hum_to_melody_v2(
    model_server: ModelServer,
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
    """
    Enhanced hum2melody endpoint with session management and visualization data.

    Args:
        audio: Audio file upload
        save_training_data: Save to database for training
        instrument: Instrument to use for playback
        onset_high: Onset detection high threshold
        onset_low: Onset detection low threshold
        offset_high: Offset detection high threshold
        offset_low: Offset detection low threshold
        min_confidence: Minimum confidence to keep notes
        return_visualization: Whether to return visualization data

    Returns:
        JSON with IR, session_id, and optional visualization data
    """
    print("[HUM2MELODY_V2] ========================================")
    print("[HUM2MELODY_V2] Enhanced endpoint called")
    print(f"[HUM2MELODY_V2]   Filename: {audio.filename}")
    print(f"[HUM2MELODY_V2]   Instrument: {instrument}")
    print(f"[HUM2MELODY_V2]   Parameters: onset_high={onset_high}, onset_low={onset_low}")
    print(f"[HUM2MELODY_V2]                 offset_high={offset_high}, offset_low={offset_low}")
    print(f"[HUM2MELODY_V2]                 min_confidence={min_confidence}")
    print(f"[HUM2MELODY_V2]   Return visualization: {return_visualization}")

    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        print(f"[HUM2MELODY_V2]   Read {len(audio_bytes)} bytes")

        # Create session
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

        print(f"[HUM2MELODY_V2]   Created session: {session_id}")

        # Get audio path
        audio_path = session_manager.get_audio_path(session_id)

        # Process audio for features
        audio_features = audio_processor.preprocess_for_hum2melody(audio_bytes)
        audio_features["audio_bytes"] = audio_bytes
        audio_features["audio_path"] = audio_path
        audio_features["instrument"] = instrument

        # Get model prediction
        melody_track = await model_server.predict_melody(audio_features)

        if not melody_track.notes:
            print("[HUM2MELODY_V2] ⚠️  No notes predicted, forcing fallback")
            audio_features.pop("audio_bytes", None)
            audio_features.pop("audio_path", None)
            melody_track = await model_server.predict_melody(audio_features)

        print(f"[HUM2MELODY_V2]   Generated {len(melody_track.notes)} notes")

        # Create IR
        ir = IR(
            metadata={
                "tempo": 120,
                "key": "Am",
                "time_signature": "4/4",
                "duration": audio_features['duration']
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
                sample_rate=audio_features['sample_rate'],
                duration=audio_features['duration'],
                metadata={'session_id': session_id, **session_metadata}
            )
            db.save_prediction(
                audio_sample_id=audio_id,
                model_type="hum2melody",
                prediction=melody_track.model_dump()
            )
            print(f"[HUM2MELODY_V2]   Saved to database: {audio_id}")

        # Build response
        response_data = {
            "status": "success",
            "ir": ir.model_dump(),
            "session_id": session_id,
            "audio_id": audio_id,
            "metadata": {
                "duration": audio_features['duration'],
                "num_notes": len(melody_track.notes),
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

        # Add visualization data if requested
        if return_visualization:
            print("[HUM2MELODY_V2]   Extracting visualization data...")
            try:
                viz_data = extract_segments_with_detection(
                    audio_path,
                    onset_high=onset_high,
                    onset_low=onset_low,
                    offset_high=offset_high,
                    offset_low=offset_low,
                    min_confidence=min_confidence
                )

                # Get waveform data
                waveform_data = session_manager.get_waveform_data(session_id, max_samples=2000)

                response_data["visualization"] = {
                    "segments": viz_data['segments'],
                    "onsets": viz_data['onsets'],
                    "offsets": viz_data['offsets'],
                    "waveform": waveform_data,
                    "parameters": viz_data['parameters']
                }

                print(f"[HUM2MELODY_V2]   Added visualization data: {len(viz_data['segments'])} segments")
            except Exception as e:
                print(f"[HUM2MELODY_V2]   ⚠️  Failed to extract visualization: {e}")
                response_data["visualization"] = None

        print("[HUM2MELODY_V2] ========================================")
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"[HUM2MELODY_V2] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


# ============================================================
# Get segments for a session
# ============================================================

async def get_segments(session_id: str):
    """
    Get current segment detection for a session.

    Returns segment data with current parameters.
    """
    print(f"[GET_SEGMENTS] Session: {session_id}")

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


# ============================================================
# Reprocess with new parameters
# ============================================================

async def reprocess_segments(
    model_server: ModelServer,
    session_id: str = Form(...),
    manual_onsets: str = Form(...),  # Required: JSON array of onset times
    manual_offsets: str = Form(...)  # Required: JSON array of offset times
):
    """
    Reprocess audio with user-provided onset/offset markers.

    Args:
        session_id: Session identifier
        manual_onsets: JSON array of onset times (required)
        manual_offsets: JSON array of offset times (required)

    Returns:
        New IR with pitch predictions for the provided segments
    """
    print("[REPROCESS] ========================================")
    print(f"[REPROCESS] Session: {session_id}")
    print(f"[REPROCESS] Manual onsets: {manual_onsets[:100]}")
    print(f"[REPROCESS] Manual offsets: {manual_offsets[:100]}")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    audio_path = session['audio_path']
    instrument = session['metadata'].get('instrument', 'piano/grand_piano_k')

    try:
        # Parse markers
        import json

        print(f"[REPROCESS] Raw manual_onsets: {repr(manual_onsets)[:200]}")
        print(f"[REPROCESS] Raw manual_offsets: {repr(manual_offsets)[:200]}")

        onset_list = json.loads(manual_onsets)
        offset_list = json.loads(manual_offsets)

        print(f"[REPROCESS] Parsed {len(onset_list)} onsets: {onset_list}")
        print(f"[REPROCESS] Parsed {len(offset_list)} offsets: {offset_list}")

        if len(onset_list) == 0 or len(offset_list) == 0:
            raise HTTPException(status_code=400, detail="Must provide at least one onset and one offset")

        # Extract segments with user's markers (no automatic detection)
        viz_data = extract_segments_with_detection(
            audio_path,
            manual_onsets=onset_list,
            manual_offsets=offset_list
        )

        print(f"[REPROCESS] Extraction returned {len(viz_data['segments'])} segments")
        for i, seg in enumerate(viz_data['segments']):
            print(f"  Segment {i}: {seg['start']:.3f}s - {seg['end']:.3f}s, pitch={seg['pitch']}, conf={seg['confidence']:.3f}")

        # Convert segments to Notes
        notes = []
        for seg in viz_data['segments']:
            notes.append(Note(
                pitch=seg['pitch'],
                start=seg['start'],
                duration=seg['duration'],
                velocity=int(seg['confidence'] * 127)
            ))

        print(f"[REPROCESS]   Generated {len(notes)} notes")

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

        print("[REPROCESS] ========================================")

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


# ============================================================
# Delete session
# ============================================================

async def delete_session(session_id: str):
    """Delete a session and clean up files."""
    print(f"[DELETE_SESSION] Session: {session_id}")

    success = session_manager.delete_session(session_id)

    if success:
        return JSONResponse(content={"status": "success", "message": "Session deleted"})
    else:
        raise HTTPException(status_code=404, detail="Session not found")
