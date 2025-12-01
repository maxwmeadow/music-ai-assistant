"""
Beatbox2Drums: CNN-based drum transcription for beatbox recordings.

Complete end-to-end pipeline using CNN onset detection and CNN drum classification.

Version 2.0.0 - CNN-Based Pipeline
Onset Detection: 91.3% F1-score
Drum Classification: 99.39% accuracy
"""

from . import models
from . import inference
from . import data

__version__ = "2.0.0"
__all__ = ['models', 'inference', 'data']
