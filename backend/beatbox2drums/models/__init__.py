"""
Beatbox2Drums CNN model architectures.
"""

from .drum_classifier import DrumClassifierCNN, create_model

__all__ = [
    'DrumClassifierCNN',
    'create_model',
]
