@echo off
setlocal
cd /d "%~dp0"
title Enterprise Document Assistant
echo.
echo ==========================================================
echo   Enterprise Document Assistant - one-click start (Windows)
echo ==========================================================
echo.

rem --- 1. Find Python (3.10 or newer) ---------------------------------
set "PY="
where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :nopython
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto :oldpython

rem --- 2. Create the private Python environment (first time only) -----
if exist ".venv\Scripts\python.exe" goto :haveenv
echo [1/3] Creating a private Python environment (about 20 seconds)...
%PY% -m venv .venv
if errorlevel 1 goto :fail
:haveenv

rem --- 3. Install the packages (first time only, 3-10 minutes) --------
if exist ".venv\.packages_ok" goto :haveparts
echo [2/3] Installing the packages. This takes 3 to 10 minutes the first time. Please wait...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo done> ".venv\.packages_ok"
:haveparts

rem --- 4. Start the app -------------------------------------------------
echo [3/3] Starting the app. Your web browser will open by itself.
echo       Keep this window open while you use the app. Close it (or press Ctrl+C) to stop.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:nopython
echo Python was not found on this computer.
echo Install Python 3.11 from https://www.python.org/downloads/ and TICK the box
echo "Add python.exe to PATH" on the first installer screen. Then double-click this file again.
goto :end

:oldpython
echo Your Python is too old. Please install Python 3.11 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then double-click this file again.
goto :end

:fail
echo.
echo Something went wrong (see the message above). Open README.md and look at the
echo Troubleshooting section, or copy the message above into a search engine.

:end
echo.
pause
