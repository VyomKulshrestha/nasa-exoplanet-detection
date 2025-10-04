#!/usr/bin/env python3
"""
Setup script for NASA Exoplanet Detection System
"""

from setuptools import setup, find_packages

setup(
    name="nasa-exoplanet-detection",
    version="1.0.0",
    description="AI/ML system for automatically detecting exoplanets from NASA space-based survey data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="NASA Exoplanet AI Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0", 
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scipy>=1.9.0",
        "lightkurve>=2.4.0",
        "astropy>=5.0.0",
        "astroquery>=0.4.6",
        "scikit-learn>=1.2.0",
        "tensorflow>=2.13.0",
        "keras>=2.13.0",
        "xgboost>=1.7.0",
        "plotly>=5.15.0",
        "ipywidgets>=8.0.0",
        "jupyter>=1.0.0",
        "tqdm>=4.64.0",
        "joblib>=1.2.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "pylint>=2.15.0",
            "mypy>=0.991",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="exoplanet detection machine learning astronomy nasa kepler tess",
    entry_points={
        "console_scripts": [
            "exoplanet-detect=main:main",
        ],
    },
)


