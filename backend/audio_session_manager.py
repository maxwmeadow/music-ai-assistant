"""
Audio Session Manager - Temporary storage for re-processing

Stores uploaded audio files with session IDs so users can re-process
with different parameters without re-uploading.
"""

import uuid
import time
from pathlib import Path
from typing import Dict, Optional, Any
import threading
import librosa
import numpy as np


class AudioSessionManager:
    """
    Manages temporary audio sessions for interactive tuning.

    Features:
    - Stores audio files with unique session IDs
    - Auto-cleanup after expiration (default: 1 hour)
    - Provides waveform data for visualization
    """

    def __init__(self, storage_dir: str = "audio_sessions", session_timeout: int = 3600):
        """
        Initialize session manager.

        Args:
            storage_dir: Directory to store session audio files
            session_timeout: Session expiration time in seconds (default: 1 hour)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.session_timeout = session_timeout

        # In-memory session metadata
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        print(f"[AudioSessionManager] Initialized")
        print(f"  Storage directory: {self.storage_dir.absolute()}")
        print(f"  Session timeout: {session_timeout}s ({session_timeout/3600:.1f}h)")

    def create_session(self, audio_bytes: bytes, filename: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new audio session.

        Args:
            audio_bytes: Audio file bytes
            filename: Original filename
            metadata: Optional metadata to store

        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())

        # Save audio file
        file_extension = Path(filename).suffix or '.wav'
        audio_path = self.storage_dir / f"{session_id}{file_extension}"

        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)

        # Store session metadata
        with self._lock:
            self.sessions[session_id] = {
                'created_at': time.time(),
                'filename': filename,
                'audio_path': str(audio_path),
                'audio_bytes_size': len(audio_bytes),
                'metadata': metadata or {}
            }

        print(f"[AudioSessionManager] Created session: {session_id}")
        print(f"  File: {filename} ({len(audio_bytes)} bytes)")
        print(f"  Path: {audio_path}")

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata or None if not found/expired
        """
        with self._lock:
            session = self.sessions.get(session_id)

            if not session:
                print(f"[AudioSessionManager] Session not found: {session_id}")
                return None

            # Check expiration
            age = time.time() - session['created_at']
            if age > self.session_timeout:
                print(f"[AudioSessionManager] Session expired: {session_id} (age: {age:.0f}s)")
                self._cleanup_session(session_id)
                return None

            return session

    def get_audio_path(self, session_id: str) -> Optional[str]:
        """
        Get audio file path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Audio file path or None if not found
        """
        session = self.get_session(session_id)
        if session:
            return session['audio_path']
        return None

    def get_waveform_data(self, session_id: str, max_samples: int = 2000) -> Optional[Dict[str, Any]]:
        """
        Get downsampled waveform data for visualization.

        Args:
            session_id: Session identifier
            max_samples: Maximum number of samples to return (for visualization)

        Returns:
            Dictionary with waveform data or None if not found
        """
        audio_path = self.get_audio_path(session_id)
        if not audio_path or not Path(audio_path).exists():
            return None

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            # Downsample for visualization
            if len(audio) > max_samples:
                # Use max pooling to preserve peaks
                chunk_size = len(audio) // max_samples
                downsampled = []
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    if len(chunk) > 0:
                        # Take max absolute value in chunk
                        downsampled.append(float(chunk[np.argmax(np.abs(chunk))]))
                waveform_samples = downsampled[:max_samples]
            else:
                waveform_samples = audio.tolist()

            return {
                'samples': waveform_samples,
                'sample_rate': sr,
                'duration': float(duration),
                'original_length': len(audio),
                'downsampled_length': len(waveform_samples)
            }

        except Exception as e:
            print(f"[AudioSessionManager] Error loading waveform: {e}")
            return None

    def _cleanup_session(self, session_id: str):
        """Clean up session files and metadata (internal, assumes lock held)."""
        session = self.sessions.get(session_id)
        if session:
            # Delete audio file
            audio_path = Path(session['audio_path'])
            if audio_path.exists():
                try:
                    audio_path.unlink()
                    print(f"[AudioSessionManager] Deleted file: {audio_path}")
                except Exception as e:
                    print(f"[AudioSessionManager] Error deleting file: {e}")

            # Remove from sessions
            del self.sessions[session_id]
            print(f"[AudioSessionManager] Cleaned up session: {session_id}")

    def cleanup_expired_sessions(self):
        """Clean up all expired sessions."""
        current_time = time.time()
        expired_sessions = []

        with self._lock:
            for session_id, session in list(self.sessions.items()):
                age = current_time - session['created_at']
                if age > self.session_timeout:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self._cleanup_session(session_id)

        if expired_sessions:
            print(f"[AudioSessionManager] Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def delete_session(self, session_id: str) -> bool:
        """
        Manually delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                self._cleanup_session(session_id)
                return True
            return False

    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self.sessions)


# Global instance
_session_manager: Optional[AudioSessionManager] = None


def get_session_manager() -> AudioSessionManager:
    """Get or create global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = AudioSessionManager()
    return _session_manager
