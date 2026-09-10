"""
Setup script for fsoc-pat-simulator backend package.
Enables editable installs via `pip install -e .` for frontend/GUI dashboard integration.
"""

from setuptools import setup, find_packages

setup(
    name="fsoc-pat-simulator",
    version="1.0.0",
    description="Free-Space Optical Communication (FSOC) Coarse Pointing-Acquisition-Tracking (PAT) Simulator",
    author="SIH 2024 PS-26169 Team (ISRO / DOS)",
    py_modules=["contracts"],
    packages=find_packages(include=[
        "sim", "sim.*",
        "detect", "detect.*",
        "track", "track.*",
        "control", "control.*",
        "disturb", "disturb.*",
        "metrics", "metrics.*",
    ]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "opencv-python>=4.5.0",
    ],
)
