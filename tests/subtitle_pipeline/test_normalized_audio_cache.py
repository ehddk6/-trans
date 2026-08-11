import json

from subtitle_pipeline.models import SourceSegment, Word
from subtitle_pipeline.pipeline import run_pipeline


def test_run_pipeline_uses_shared_normalized_audio_but_keeps_original_review_media(monkeypatch, tmp_path):
    media = tmp_path / "TEST-001.mp4"
    media.write_bytes(b"video-placeholder")
    normalized = tmp_path / "cached.wav"
    normalized.write_bytes(b"wav-placeholder")
    captured = {}

    def fake_transcribe(input_path, *args, **kwargs):
        captured["recognition_input"] = input_path
        return [
            SourceSegment(
                id=0,
                start=0.0,
                end=1.5,
                text="こんにちは",
                words=[Word("こんにちは", 0.0, 1.5, 0.99, backend="whisper")],
            )
        ], 1.5

    monkeypatch.setattr("subtitle_pipeline.pipeline.transcribe", fake_transcribe)
    output = tmp_path / "output"
    run_pipeline(
        media,
        output,
        backend="faster-whisper",
        normalized_audio_path=normalized,
        review_samples=1,
    )

    assert captured["recognition_input"] == normalized
    manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_path"] == str(media.resolve())
    assert manifest["video_path"] != str(normalized.resolve())
