"""
python/db.py - Correction Database
All personal user data. Never touches the app install directory.
DB lives at ~/.voiceadapt/corrections.db
"""

import sqlite3, time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from rich.console import Console

console = Console()
DATA_DIR = Path.home() / '.voiceadapt'
DB_PATH  = DATA_DIR / 'corrections.db'


@dataclass
class CorrectionPair:
    id: Optional[int]
    chunk_uuid: str
    audio_path: Optional[str]
    raw_transcript: str
    corrected_text: str
    adapter_ver: str
    timestamp: float
    trained: bool = False

    @property
    def was_corrected(self) -> bool:
        return self.raw_transcript.strip() != self.corrected_text.strip()


class CorrectionDB:

    def __init__(self, db_path=DB_PATH):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn   = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        self._conn.executescript('''
            CREATE TABLE IF NOT EXISTS correction_pairs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_uuid     TEXT NOT NULL,
                audio_path     TEXT,
                raw_transcript TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                adapter_ver    TEXT NOT NULL DEFAULT 'base',
                timestamp      REAL NOT NULL,
                trained        INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_trained ON correction_pairs(trained);
            CREATE TABLE IF NOT EXISTS vocab_frequency (
                word       TEXT PRIMARY KEY,
                count      INTEGER NOT NULL DEFAULT 0,
                first_seen REAL NOT NULL,
                last_seen  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS adapter_versions (
                version    TEXT PRIMARY KEY,
                path       TEXT NOT NULL,
                trained_on INTEGER NOT NULL,
                wer_before REAL,
                wer_after  REAL,
                created_at REAL NOT NULL
            );
        ''')
        self._conn.commit()

    def insert_pair(self, chunk_uuid, raw_transcript, corrected_text,
                    adapter_ver='base', audio_path=None) -> int:
        cur = self._conn.execute(
            'INSERT INTO correction_pairs (chunk_uuid,audio_path,raw_transcript,'
            'corrected_text,adapter_ver,timestamp) VALUES (?,?,?,?,?,?)',
            (chunk_uuid, audio_path, raw_transcript, corrected_text, adapter_ver, time.time())
        )
        self._conn.commit()
        self._update_vocab(corrected_text)
        return cur.lastrowid

    def count_untrained_pairs(self) -> int:
        return self._conn.execute(
            'SELECT COUNT(*) FROM correction_pairs WHERE trained=0'
        ).fetchone()[0]

    def get_untrained_pairs(self, limit=500):
        rows = self._conn.execute(
            'SELECT * FROM correction_pairs WHERE trained=0 ORDER BY timestamp ASC LIMIT ?',
            (limit,)
        ).fetchall()
        return [CorrectionPair(
            id=r['id'], chunk_uuid=r['chunk_uuid'], audio_path=r['audio_path'],
            raw_transcript=r['raw_transcript'], corrected_text=r['corrected_text'],
            adapter_ver=r['adapter_ver'], timestamp=r['timestamp'], trained=bool(r['trained'])
        ) for r in rows]

    def mark_pairs_trained(self, pair_ids: list):
        ph = ','.join('?'*len(pair_ids))
        self._conn.execute(f'UPDATE correction_pairs SET trained=1 WHERE id IN ({ph})', pair_ids)
        self._conn.commit()

    def _update_vocab(self, text: str):
        now   = time.time()
        words = [w.lower().strip('.,!?;:\'"') for w in text.split() if len(w) > 2]
        for word in words:
            self._conn.execute(
                'INSERT INTO vocab_frequency (word,count,first_seen,last_seen) VALUES (?,1,?,?)'
                ' ON CONFLICT(word) DO UPDATE SET count=count+1, last_seen=excluded.last_seen',
                (word, now, now)
            )
        self._conn.commit()

    def get_top_vocab(self, limit=50):
        rows = self._conn.execute(
            'SELECT word,count FROM vocab_frequency ORDER BY count DESC LIMIT ?', (limit,)
        ).fetchall()
        return [{'word':r['word'],'count':r['count']} for r in rows]

    def log_adapter_version(self, version, path, trained_on, wer_before=None, wer_after=None):
        self._conn.execute(
            'INSERT OR REPLACE INTO adapter_versions (version,path,trained_on,wer_before,wer_after,created_at)'
            ' VALUES (?,?,?,?,?,?)', (version, path, trained_on, wer_before, wer_after, time.time())
        )
        self._conn.commit()

    def stats(self) -> dict:
        total     = self._conn.execute('SELECT COUNT(*) FROM correction_pairs').fetchone()[0]
        untrained = self.count_untrained_pairs()
        corrected = self._conn.execute(
            'SELECT COUNT(*) FROM correction_pairs WHERE raw_transcript!=corrected_text'
        ).fetchone()[0]
        vocab = self._conn.execute('SELECT COUNT(*) FROM vocab_frequency').fetchone()[0]
        return {'total_pairs':total,'untrained_pairs':untrained,'corrected_pairs':corrected,'vocab_size':vocab}

    def close(self):
        self._conn.close()
