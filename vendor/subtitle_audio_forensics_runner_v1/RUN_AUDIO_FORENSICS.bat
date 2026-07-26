@echo off
setlocal
set ROOT=%~dp0
set PROJECT=%~1
if "%PROJECT%"=="" set PROJECT=%CD%

set PYTHON=python
if exist "%USERPROFILE%\.venv\Scripts\python.exe" set PYTHON=%USERPROFILE%\.venv\Scripts\python.exe
if exist "%PROJECT%\.venv\Scripts\python.exe" set PYTHON=%PROJECT%\.venv\Scripts\python.exe

"%PYTHON%" "%ROOT%run_forensic_audio.py" --project-dir "%PROJECT%" --bands P1,P2
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" echo 실행 실패. 위 오류를 확인하세요.
if "%ERR%"=="0" echo 결과 ZIP이 작품 폴더에 생성되었습니다.
pause
exit /b %ERR%
