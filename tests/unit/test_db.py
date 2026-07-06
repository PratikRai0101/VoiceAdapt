import pytest
from python.db import CorrectionDB

@pytest.fixture
def db(tmp_path):
    db = CorrectionDB(db_path=tmp_path / 'test.db')
    yield db
    db.close()

class TestInsertPair:
    def test_returns_id(self, db):
        assert db.insert_pair('u1','raw','corrected') == 1
    def test_ids_increment(self, db):
        id1 = db.insert_pair('u1','raw','corrected')
        id2 = db.insert_pair('u2','raw','corrected')
        assert id2 == id1 + 1
    def test_count_increases(self, db):
        db.insert_pair('u1','a','b'); db.insert_pair('u2','c','d')
        assert db.count_untrained_pairs() == 2

class TestVocab:
    def test_vocab_tracked(self, db):
        db.insert_pair('u1','x','Hyprland window manager')
        vocab = {v['word'] for v in db.get_top_vocab()}
        assert 'hyprland' in vocab and 'window' in vocab
    def test_count_increments(self, db):
        db.insert_pair('u1','x','zkSync is fast')
        db.insert_pair('u2','x','zkSync is great')
        vocab = {v['word']:v['count'] for v in db.get_top_vocab()}
        assert vocab['zksync'] == 2
    def test_short_words_excluded(self, db):
        db.insert_pair('u1','x','I am at it')
        vocab = {v['word'] for v in db.get_top_vocab()}
        assert 'i' not in vocab and 'am' not in vocab

class TestMarkTrained:
    def test_mark_reduces_count(self, db):
        id1 = db.insert_pair('u1','a','b')
        db.mark_pairs_trained([id1])
        assert db.count_untrained_pairs() == 0
    def test_trained_not_returned(self, db):
        id1 = db.insert_pair('u1','a','b')
        db.mark_pairs_trained([id1])
        assert db.get_untrained_pairs() == []

class TestStats:
    def test_stats_keys_exist(self, db):
        stats = db.stats()
        assert 'total_pairs' in stats and 'vocab_size' in stats
