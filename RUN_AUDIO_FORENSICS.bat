@echo off
setlocal
set "ROOT=%~dp0"
set "PROJECT=%~1"
if "%PROJECT%"=="" set "PROJECT=%CD%"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
for %%I in ("%PROJECT%") do set "TITLE=%%~nxI"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
"%PYTHON%" -m translation_forensics.cli prepare-audio --project-root "%ROOT%" --title "%TITLE%" --workspace "%PROJECT%" --json
exit /b %ERRORLEVEL%
