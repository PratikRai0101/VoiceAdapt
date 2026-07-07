"""
Integration tests for the full VoiceAdapt pipeline.
Requires: Whisper model (downloaded once), MPS or CPU.
Run: pytest tests/integration/ -v
"""

import soundfile as sf
import pytest
from pathlib import Path
from python.vad import AudioChunk, SAMPLE_RATE
from python.inference import InferenceEngine
from python.db import CorrectionDB

FIXTURE_DIR = Path(__file__).parent.parent / 'fixtures'


@pytest.fixture(scope='module')
def engine():
    try:
        e = InferenceEngine(device='mps')
        e.load()
        return e
    except Exception as exc:
        pytest.skip(f'InferenceEngine unavailable: {exc}')


@pytest.fixture
def db(tmp_path):
    db = CorrectionDB(db_path=tmp_path / 'test.db')
    yield db
    db.close()


class TestFullPipeline:

    def test_transcribe_fixture(self, engine):
        audio, sr = sf.read(str(FIXTURE_DIR / 'test_silence.wav'), dtype='float32')
        chunk = AudioChunk(audio=audio)
        result = engine.transcribe(chunk)
        assert result.chunk_uuid == chunk.uuid
        assert result.language == 'en'
        assert result.latency_ms > 0

    def test_chunk_uuid_links_through_pipeline(self, engine, db):
        audio, sr = sf.read(str(FIXTURE_DIR / 'test_silence.wav'), dtype='float32')
        chunk = AudioChunk(audio=audio)
        result = engine.transcribe(chunk)
        pair_id = db.insert_pair(
            chunk_uuid=chunk.uuid,
            raw_transcript=result.text,
            corrected_text=result.text,
        )
        pairs = db.get_untrained_pairs()
        assert len(pairs) == 1
        assert pairs[0].chunk_uuid == chunk.uuid
        assert pairs[0].raw_transcript == result.text

    def test_correction_stored_separately(self, engine, db):
        audio, sr = sf.read(str(FIXTURE_DIR / 'test_silence.wav'), dtype='float32')
        chunk = AudioChunk(audio=audio)
        result = engine.transcribe(chunk)
        corrected = 'this is the corrected version'
        db.insert_pair(
            chunk_uuid=chunk.uuid,
            raw_transcript=result.text,
            corrected_text=corrected,
        )
        pairs = db.get_untrained_pairs()
        assert pairs[0].was_corrected
        assert pairs[0].corrected_text == corrected
