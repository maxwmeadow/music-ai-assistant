import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

class TrainingDataDB:
    """
    SQLite database for storing training data (audio files and metadata).
    Supports recording user interactions for future model training.
    """
    
    def __init__(self, db_path: str = "training_data.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Audio samples table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_format TEXT NOT NULL,
                    sample_rate INTEGER,
                    duration REAL,
                    model_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metadata table (flexible JSON storage)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_sample_id INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (audio_sample_id) REFERENCES audio_samples(id)
                )
            ''')
            
            # Model predictions table (store model outputs for evaluation)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_sample_id INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    prediction_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_sample_id) REFERENCES audio_samples(id)
                )
            ''')
            
            # User feedback table (for future model improvements)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_sample_id INTEGER NOT NULL,
                    prediction_id INTEGER,
                    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_sample_id) REFERENCES audio_samples(id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def save_audio_sample(
        self, 
        file_path: str,
        model_type: str,
        file_format: str = "wav",
        sample_rate: Optional[int] = None,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save audio sample record to database.
        
        Args:
            file_path: Path where audio file is stored
            model_type: Type of model this audio is for ("hum2melody", "beatbox2drums", etc)
            file_format: Audio file format (wav, mp3, etc)
            sample_rate: Audio sample rate in Hz
            duration: Audio duration in seconds
            metadata: Additional metadata as dictionary
            
        Returns:
            Database ID of inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert audio sample
            cursor.execute('''
                INSERT INTO audio_samples (file_path, file_format, sample_rate, duration, model_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_path, file_format, sample_rate, duration, model_type))
            
            audio_id = cursor.lastrowid
            if audio_id is None:
                raise RuntimeError("Failed to insert audio sample: no ID returned")
            
            # Insert metadata if provided
            if metadata:
                metadata_json = json.dumps(metadata)
                cursor.execute('''
                    INSERT INTO audio_metadata (audio_sample_id, metadata_json)
                    VALUES (?, ?)
                ''', (audio_id, metadata_json))
            
            conn.commit()
            return audio_id
    
    def save_prediction(
        self,
        audio_sample_id: int,
        model_type: str,
        prediction: Dict[str, Any]
    ) -> int:
        """
        Save model prediction to database.
        
        Args:
            audio_sample_id: ID of audio sample this prediction is for
            model_type: Type of model that made prediction
            prediction: Prediction data as dictionary
            
        Returns:
            Database ID of inserted prediction
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            prediction_json = json.dumps(prediction)
            cursor.execute('''
                INSERT INTO predictions (audio_sample_id, model_type, prediction_json)
                VALUES (?, ?, ?)
            ''', (audio_sample_id, model_type, prediction_json))
            
            prediction_id = cursor.lastrowid
            if prediction_id is None:
                raise RuntimeError("Failed to insert prediction: no ID returned")
            
            conn.commit()
            return prediction_id
    
    def save_feedback(
        self,
        audio_sample_id: int,
        rating: int,
        prediction_id: Optional[int] = None,
        feedback_text: Optional[str] = None
    ) -> int:
        """
        Save user feedback for a prediction.
        
        Args:
            audio_sample_id: ID of audio sample
            rating: User rating (1-5)
            prediction_id: Optional ID of specific prediction
            feedback_text: Optional text feedback
            
        Returns:
            Database ID of inserted feedback
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_feedback (audio_sample_id, prediction_id, rating, feedback_text)
                VALUES (?, ?, ?, ?)
            ''', (audio_sample_id, prediction_id, rating, feedback_text))
            
            feedback_id = cursor.lastrowid
            if feedback_id is None:
                raise RuntimeError("Failed to insert feedback: no ID returned")
            
            conn.commit()
            return feedback_id
    
    def get_audio_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve audio sample by ID.
        
        Args:
            sample_id: Database ID of audio sample
            
        Returns:
            Dictionary with audio sample data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM audio_samples WHERE id = ?
            ''', (sample_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_samples_by_model_type(
        self, 
        model_type: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all samples for a specific model type.
        
        Args:
            model_type: Model type to filter by
            limit: Maximum number of results
            
        Returns:
            List of audio sample records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM audio_samples 
                WHERE model_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (model_type, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_training_dataset(
        self, 
        model_type: str,
        min_rating: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training dataset with audio samples, predictions, and feedback.
        
        Args:
            model_type: Model type to get data for
            min_rating: Optional minimum user rating filter
            
        Returns:
            List of training data records with audio, predictions, and feedback
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    a.id as audio_id,
                    a.file_path,
                    a.duration,
                    a.sample_rate,
                    p.prediction_json,
                    f.rating,
                    f.feedback_text
                FROM audio_samples a
                LEFT JOIN predictions p ON a.id = p.audio_sample_id
                LEFT JOIN user_feedback f ON a.id = f.audio_sample_id
                WHERE a.model_type = ?
            '''
            
            params: List[Any] = [model_type]
            
            if min_rating is not None:
                query += ' AND (f.rating >= ? OR f.rating IS NULL)'
                params.append(min_rating)
            
            query += ' ORDER BY a.created_at DESC'
            
            cursor.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with various statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total samples by model type
            cursor.execute('''
                SELECT model_type, COUNT(*) as count
                FROM audio_samples
                GROUP BY model_type
            ''')
            stats['samples_by_type'] = {row['model_type']: row['count'] for row in cursor.fetchall()}
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) as count FROM predictions')
            stats['total_predictions'] = cursor.fetchone()['count']
            
            # Average rating
            cursor.execute('SELECT AVG(rating) as avg_rating FROM user_feedback')
            avg_rating = cursor.fetchone()['avg_rating']
            stats['average_rating'] = round(avg_rating, 2) if avg_rating else None
            
            # Total feedback entries
            cursor.execute('SELECT COUNT(*) as count FROM user_feedback')
            stats['total_feedback'] = cursor.fetchone()['count']
            
            return stats