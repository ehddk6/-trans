$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    python -m venv $Venv
}
& $Python -m pip install --upgrade pip
& $Python -m pip install "faster-whisper"
Write-Host "ASR 설치 완료: $Venv"
Write-Host "ffmpeg -version 명령도 정상 작동하는지 확인하세요."
