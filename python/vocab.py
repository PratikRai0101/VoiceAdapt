"""
python/vocab.py - Vocabulary Viewer
Shows your most frequently used words from correction history.
Usage: python3 -m python.vocab
"""

from rich.console import Console
from rich.table import Table
from python.db import CorrectionDB

console = Console()


def show_vocab(limit=50):
    db = CorrectionDB()
    vocab = db.get_top_vocab(limit)
    if not vocab:
        console.print('[yellow]No vocabulary data yet. Start using VoiceAdapt![/yellow]')
        return
    t = Table(show_header=True, header_style='bold blue')
    t.add_column('#', style='dim')
    t.add_column('Word')
    t.add_column('Count', justify='right')
    for i, entry in enumerate(vocab, 1):
        t.add_row(str(i), entry['word'], str(entry['count']))
    console.print(t)
    console.print(f'[dim]Total unique words: {len(vocab)}[/dim]')
    db.close()


if __name__ == '__main__':
    show_vocab()
