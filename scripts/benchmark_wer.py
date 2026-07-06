"""
scripts/benchmark_wer.py - WER Baseline Measurement
Usage: python3 scripts/benchmark_wer.py --fixtures tests/fixtures/
"""

import json, argparse
from pathlib import Path
from jiwer import wer
from python.inference import InferenceEngine
from python.vad import AudioChunk
import soundfile as sf

# Your reference sentences - what you actually said
SENTENCES = [
    'My name is Pratik Rai',
    'I work at DIMO on zkSync smart contracts',
    'Hyprland is my window manager on Arch Linux',
    'YCCE is my college in Nagpur',
    'NousResearch builds open source language models',
]


def run_benchmark(audio_dir: str, adapter_ver: str = 'base'):
    engine = InferenceEngine(device='mps')
    results = []
    audio_files = sorted(Path(audio_dir).glob('*.wav'))
    for i, (audio_file, reference) in enumerate(zip(audio_files, SENTENCES)):
        audio, sr = sf.read(str(audio_file), dtype='float32')
        if audio.ndim > 1: audio = audio[:,0]
        chunk = AudioChunk(audio=audio)
        result = engine.transcribe(chunk)
        error = wer(reference.lower(), result.text.lower())
        results.append({'reference': reference, 'hypothesis': result.text,
                        'wer': error, 'latency_ms': result.latency_ms})
        print(f'[{i+1}/{len(SENTENCES)}] WER: {error:.3f}  |  {result.text}')
    avg_wer = sum(r['wer'] for r in results) / len(results)
    output  = {'adapter_ver': adapter_ver, 'avg_wer': avg_wer, 'results': results}
    out_path = f'tests/fixtures/wer_{adapter_ver}.json'
    with open(out_path, 'w') as f: json.dump(output, f, indent=2)
    print(f'\nAverage WER: {avg_wer:.3f} - saved to {out_path}')
    return avg_wer


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fixtures', default='tests/fixtures/')
    p.add_argument('--adapter',  default='base')
    args = p.parse_args()
    run_benchmark(args.fixtures, args.adapter)
