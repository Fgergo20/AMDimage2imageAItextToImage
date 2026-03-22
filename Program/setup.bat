@echo off
echo ==========================================
echo  AI Image Generator - FULL Setup
echo ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Install Python 3.10 or 3.11 and add to PATH.
    pause
    exit /b 1
)

:: Create venv
echo Creating virtual environment...
python -m venv venv

:: Activate venv
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch (DirectML for AMD / fallback CPU)
echo Installing PyTorch (DirectML)...
pip install torch-directml==0.2.5.dev240914

:: Core AI libraries
echo Installing core AI libraries...
pip install ^
diffusers==0.26.3 ^
transformers==4.36.0 ^
accelerate==0.26.0 ^
huggingface-hub==0.23.0 ^
safetensors==0.4.2

:: Image + processing
echo Installing image libraries...
pip install ^
pillow ^
numpy ^
opencv-python ^
scipy

:: Upscaling stack (Real-ESRGAN)
echo Installing upscaling libraries...
pip install ^
realesrgan==0.3.0 ^
basicsr ^
facexlib

:: Utilities
echo Installing utilities...
pip install ^
tqdm ^
requests

echo.
echo ==========================================
echo  Setup COMPLETE
echo ==========================================
echo.
echo To run:
echo   venv\Scripts\activate
echo   python img2img_ui.py
echo.
pause