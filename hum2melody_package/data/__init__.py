"""
Hum2Melody Data Package

Dataset classes and data generation utilities.
"""

from .melody_dataset import MelodyDataset, EnhancedMelodyDataset
from .synthetic_data_generator import SyntheticMelodyGenerator

__all__ = [
    'MelodyDataset',
    'EnhancedMelodyDataset',
    'SyntheticMelodyGenerator',
]
