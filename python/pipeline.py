"""
python/pipeline.py - Main Pipeline
Connects: VAD → Inference → Correction UI → DB
"""

from rich.console import Console
from rich.table import Table
from python.vad       import VADSegmenter, segment_file, AudioChunk
from python.inference import InferenceEngine
from python.db        import CorrectionDB

console           = Console()
FINE_TUNE_THRESHOLD = 50


class VoiceAdaptPipeline:

    def __init__(self, device='mps'):
        self.engine    = InferenceEngine(device=device)
        self.db        = CorrectionDB()
        self.segmenter = VADSegmenter()

    def run_mic(self):
        console.rule('[bold blue]VoiceAdapt[/bold blue]')
        self._print_stats()
        try:
            for chunk in self.segmenter.stream_mic():
                self._process(chunk)
        except KeyboardInterrupt:
            console.print('[bold red]Stopped.[/bold red]')
            self._print_stats()

    def run_file(self, path: str):
        for chunk in segment_file(path):
            self._process(chunk, interactive=False)
        self._print_stats()

    def _process(self, chunk: AudioChunk, interactive=True):
        result = self.engine.transcribe(chunk)
        if not result.text: return
        corrected = self._get_correction(result.text) if interactive else result.text
        self.db.insert_pair(chunk_uuid=chunk.uuid, raw_transcript=result.text,
                            corrected_text=corrected, adapter_ver=result.adapter_ver)
        if self.db.count_untrained_pairs() >= FINE_TUNE_THRESHOLD:
            console.print('[cyan] Fine-tune threshold reached (Sprint 2 will train here)[/cyan]')

    def _get_correction(self, raw: str) -> str:
        console.print(f'[yellow]Heard:[/yellow] {raw}')
        console.print('[dim]Enter to accept, or type correction:[/dim] ', end='')
        try:
            inp = input().strip()
            return inp if inp else raw
        except EOFError:
            return raw

    def _print_stats(self):
        stats = self.db.stats()
        t = Table(show_header=True, header_style='bold blue')
        t.add_column('Metric'); t.add_column('Value', justify='right')
        for k,v in stats.items():
            t.add_row(k.replace('_',' ').title(), str(v))
        console.print(t)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--file',   type=str)
    p.add_argument('--device', type=str, default='mps')
    args = p.parse_args()
    pipeline = VoiceAdaptPipeline(device=args.device)
    if args.file: pipeline.run_file(args.file)
    else:         pipeline.run_mic()
