from subtitle_pipeline.srt import read_srt_rows


SRT_TEXT = "1\n00:00:01,000 --> 00:00:02,000\n日本語の台詞です。\n"


def test_read_srt_rows_decodes_japanese_cp932(tmp_path):
    path = tmp_path / "cp932.srt"
    path.write_bytes(SRT_TEXT.encode("cp932"))

    assert read_srt_rows(path) == [
        {"id": "1", "start": 1.0, "end": 2.0, "text": "日本語の台詞です。"},
    ]


def test_read_srt_rows_decodes_bom_utf16(tmp_path):
    path = tmp_path / "utf16.srt"
    path.write_bytes(SRT_TEXT.encode("utf-16"))

    assert read_srt_rows(path) == [
        {"id": "1", "start": 1.0, "end": 2.0, "text": "日本語の台詞です。"},
    ]
