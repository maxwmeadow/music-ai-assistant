"""
Beatbox2Drums inference module (CNN-based).

Complete end-to-end pipeline using CNN onset detection and CNN drum classification.
"""

# Import CNN onset detector
try:
    from .cnn_onset_detector import CNNOnsetDetector
    _has_onset_detector = True
except ImportError:
    _has_onset_detector = False
    CNNOnsetDetector = None

# Import complete pipeline
try:
    from .beatbox2drums_pipeline import Beatbox2DrumsPipeline, DrumHit
    _has_pipeline = True
except ImportError:
    _has_pipeline = False
    Beatbox2DrumsPipeline = None
    DrumHit = None

__all__ = []

if _has_onset_detector:
    __all__.append('CNNOnsetDetector')

if _has_pipeline:
    __all__.extend(['Beatbox2DrumsPipeline', 'DrumHit'])
