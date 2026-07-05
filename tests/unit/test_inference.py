from python.inference import TranscriptResult


class TestTranscriptResult:
    def test_word_count(self):
        result = TranscriptResult(
            text="hello world test", chunk_uuid="x", language="en", latency_ms=100.0
        )
        assert result.word_count == 3

    def test_empty_text_word_count(self):
        result = TranscriptResult(
            text="", chunk_uuid="x", language="en", latency_ms=50.0
        )
        assert result.word_count == 0

    def test_default_adapter_version(self):
        result = TranscriptResult(
            text="hi", chunk_uuid="x", language="en", latency_ms=50.0
        )
        assert result.adapter_ver == "base"
