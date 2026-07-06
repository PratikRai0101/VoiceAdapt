import numpy as np
import pytest

from python.vad import SAMPLE_RATE, AudioChunk


class TestAudioChunk:
    def test_uuid_auto_generated(self):
        chunk = AudioChunk(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert chunk.uuid is not None and len(chunk.uuid) == 36

    def test_two_chunks_have_different_uuids(self):
        chunk1 = AudioChunk(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
        chunk2 = AudioChunk(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert chunk1.uuid != chunk2.uuid

    def test_duration_ms_one_second(self):
        chunk = AudioChunk(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert abs(chunk.duration_ms - 1000.0) < 1.0

    def test_duration_s_three_seconds(self):
        chunk = AudioChunk(audio=np.zeros(3 * SAMPLE_RATE, dtype=np.float32))
        assert abs(chunk.duration_s - 3.0) < 1.0

    def test_audio_is_preserved(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        chunk = AudioChunk(audio=audio)
        np.testing.assert_array_equal(chunk.audio, audio)

    def test_timestamp_is_positive(self):
        chunk = AudioChunk(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert chunk.timestamp > 0
