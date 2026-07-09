@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [Setup] Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    for /f "usebackq delims=" %%P in (`uv python find 3.11 2^>nul`) do set "PY311=%%P"
    if defined PY311 (
      "!PY311!" -m venv .venv
    ) else (
      echo [Error] Python 3.11 is required. Please install Python 3.11 and try again.
      pause
      exit /b 1
    )
  )
  if not exist "%VENV_PY%" (
    echo [Error] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo [Setup] Installing/updating dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [Error] Failed to upgrade pip.
  pause
  exit /b 1
)

"%VENV_PY%" -m pip install -r "%~dp0backend\requirements.txt"
if errorlevel 1 (
  echo [Error] Failed to install dependencies.
  pause
  exit /b 1
)

echo ========================================
echo   Anima Tagger
echo   URL: http://127.0.0.1:7860
echo   Ctrl+C to stop
echo ========================================
echo.

start "" http://127.0.0.1:7860

"%VENV_PY%" -m uvicorn main:app --app-dir "%~dp0backend" --host 127.0.0.1 --port 7860

echo.
pause
