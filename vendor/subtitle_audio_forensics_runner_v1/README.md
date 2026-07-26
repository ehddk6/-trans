# Subtitle Audio Forensics Runner v1

작품별 Subtitle Forensics `review-queue`와 원음에서 P1·P2 장면만 추출해 적응형 `faster-whisper` 교차인식을 수행하고 `<작품명>.work_audio.zip`을 만든다.

## 최초 1회 설치

PowerShell에서 이 폴더로 이동한 뒤:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\INSTALL_ASR.ps1
```

`ffmpeg -version`도 정상 작동해야 한다.

## 작품 폴더 준비

같은 작품 폴더 안에 최소한 다음 두 파일이 있어야 한다.

- `*.review-queue*.csv`
- MP3 또는 영상 파일

review queue 필수 열은 `block_number`, `timecode`, `review_band`다.

## 실행

작품 폴더를 배치 파일 위로 끌어다 놓거나, PowerShell에서:

```powershell
.\RUN_AUDIO_FORENSICS.bat "D:\작품폴더"
```

후보 파일이 하나씩이면 자동 선택한다. review queue나 원음 후보가 여러 개면 임의로 고르지 않고 중단하므로 Python 명령으로 정확히 지정한다.

```powershell
& "$HOME\.venv\Scripts\python.exe" .\run_forensic_audio.py `
  --project-dir "D:\작품폴더" `
  --queue "D:\작품폴더\작품.review-queue.audio-pending-v1.csv" `
  --audio "D:\작품폴더\작품.mp3"
```

기본 ZIP에는 CSV·JSON만 들어가며 WAV 조각은 제외된다. 사람이 직접 청취할 자료도 보관하려면 `--include-clips`를 추가한다.

## 출력

작품 폴더에 다음이 생성된다.

- `<작품명>.work_audio/manifest.json`
- `<작품명>.work_audio/review-scenes.csv`
- `<작품명>.work_audio/asr-candidates.csv`
- `<작품명>.work_audio/run-summary.json`
- `<작품명>.work_audio.zip`

실행을 중단해도 `asr-candidates.csv`에 완료한 패스가 저장되며, 다시 실행하면 완료분을 건너뛴다.

## 판정 한계

이 실행 결과는 `audio-asr-crosschecked`용 입력이다. 사람이 일본어 원음을 직접 확인한 `audio-human-verified` 결과가 아니다. 같은 Whisper 모델의 여러 패스는 완전히 독립된 증거가 아니다.
