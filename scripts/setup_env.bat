@echo off
setlocal
echo ========================================================
echo   Layeris Python Environment Setup
echo ========================================================

echo [*] Checking Python installation...
py -3 --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3 not found in PATH! Please install Python 3.9+ first.
        pause
        exit /b 1
    )
    set PY_CMD=python
) else (
    set PY_CMD=py -3
)

echo [*] Creating virtual environment (.venv)...
%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo [*] Installing package in editable mode with development dependencies...
python -m pip install -e ".[dev]"

echo.
echo ========================================================
echo   Setup Complete!
echo.
echo   To activate the environment:
echo     .venv\Scripts\activate.bat
echo.
echo   To run tests:
echo     python -m unittest discover -s tests -p "test_*.py"
echo.
echo   To run simulation example:
echo     python examples\01_prefetch_simulation.py
echo ========================================================
pause
