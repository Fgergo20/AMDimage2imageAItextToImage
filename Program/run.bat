@echo off
echo "=========================================="
echo "AI Image Generator & Stylizer"
echo "=========================================="
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate

:: Run the application
python img2img_ui.py

:: If the script exits, keep the window open to see any error messages
echo.
echo Application closed.
pause