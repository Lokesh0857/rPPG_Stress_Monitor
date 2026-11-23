@echo off
title rPPG Advanced Dashboard - Repair and Run
echo.
echo [1/3] Removing incompatible packages...
python -m pip uninstall -y mediapipe numpy
echo.
echo [2/3] Installing compatible packages...
python -m pip install --upgrade pip
python -m pip install "numpy>=1.24,<2.0" "mediapipe==0.10.21" "scipy>=1.10" "opencv-python>=4.8" "Pillow>=10.0" "matplotlib>=3.7"
echo.
echo [3/3] Starting application...
python main.py
pause
