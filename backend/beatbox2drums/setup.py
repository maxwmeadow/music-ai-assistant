"""
Setup script for Beatbox2Drums package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "Beatbox2Drums: Production-ready drum classification for beatbox recordings"

setup(
    name="beatbox2drums",
    version="1.0.0",
    description="CNN-based drum classification system for beatbox and vocal percussion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Claude Code AI",
    author_email="noreply@anthropic.com",
    url="https://github.com/yourusername/beatbox2drums",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "librosa>=0.9.0",
        "soundfile>=0.11.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="beatbox drums percussion classification deep-learning audio music",
    package_data={
        "": ["checkpoints/*.pth"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "beatbox2drums=examples.basic_inference:main",
        ],
    },
)
