$ErrorActionPreference = "Stop"
$Venv = Join-Path $HOME ".venv"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    python -m venv $Venv
}
& (Join-Path $Venv "Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $Venv "Scripts\python.exe") -m pip install faster-whisper
Write-Host "설치 완료: $Venv"
Write-Host "ffmpeg -version 명령도 정상 작동하는지 확인하세요."
