import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
from mpmath.function_docs import si
from numpy.random.mtrand import f
from rich import inspect
from rich.abc import t
from rich.console import Console
from torch import onnx
from torch._refs import abs_

console = Console()

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SAMPLES = int(CHUNK_DURATION_MS * SAMPLE_RATE // 1000)
MIN_SPEECH_MS = 250
MAX_SILENCE_MS = 800
MAX_UTTERANCE_S = 30


@dataclass
class AudioChunk:
    audio: np.ndarray
    sample_rate: int = SAMPLE_RATE
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    @property
    def duration_ms(self) -> float:
        return len(self.audio) / self.sample_rate * 1000

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


class VADSegmenter:
    def __init__(self, device=None, threshold=0.5):
        self.device = device
        self.threshold = threshold
        self._model = None
        self._raw_q = queue.Queue()

    def _load_model(self):
        if self._model:
            return
        console.print("[dim] Loading Silero VAD model...[/dim]")
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model = model
        console.print("[green] Silero VAD ready.[/green]")

    def _speech_prob(self, frame: np.ndarray) -> float:
        import torch

        tensor = torch.from_numpy(frame).unsqueeze(0)
        with torch.no_grad():
            return self._model(tensor, SAMPLE_RATE).item()

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            console.print(f"[yellow] {status} [/yellow]")
        self._raw_q.put(indata[:, 0].copy().astype(np.float32))

    def stream_mic(self) -> Generator[AudioChunk, None, None]:
        import sounddevice as sd

        self._load_model()
        utterance_buf, silence_frames, in_speech = [], 0, False
        silence_limit = int(MAX_SILENCE_MS / CHUNK_DURATION_MS)
        max_frames = int(MAX_UTTERANCE_S * 1000 / CHUNK_DURATION_MS)
        console.print(f"[bold green] Listening... [/bold green] Ctrl+C to stop ")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.device,
            callback=self._mic_callback,
        ):
            while True:
                try:
                    frame = self._raw_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                is_speech = self._speech_prob(frame) >= self.threshold
                if is_speech:
                    utterance_buf.append(frame)
                    silence_frames, in_speech = 0, True
                elif in_speech:
                    utterance_buf.append(frame)
                    silence_frames += 1
                    if len(utterance_buf) >= max_frames:
                        yield self._emit(utterance_buf)
                        utterance_buf, silence_frames, in_speech = [], 0, False
                    elif silence_frames >= silence_limit:
                        if (
                            np.concatenate(utterance_buf).shape[0] / SAMPLE_RATE * 1000
                            >= MIN_SPEECH_MS
                        ):
                            yield self._emit(utterance_buf)
                            utterance_buf, silence_frames, in_speech = [], 0, False

    def _emit(self, frames) -> AudioChunk:
        audio = np.concatenate(frames)
        chunk = AudioChunk(audio=audio)
        console.print(
            f"[cyan]chunk[/cyan] {chunk.uuid[:8]} [dim]{chunk.duration_s:.2f}s[/dim]"
        )
        return chunk

    def segement_file(audio_path: str, threshold=0.5) -> list:
        import soundfile as sf
        import torch

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            import torchaudio

            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio = (
                torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
                .squeeze(0)
                .numpy()
            )
            model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False
            )
            get_ts = utils[0]
            timestamps = get_ts(
                torch.from_numpy(audio),
                model,
                sampling_rate=SAMPLE_RATE,
                threshold=threshold,
                min_speech_duration_ms=MIN_SPEECH_MS,
                max_speech_duration_s=MAX_UTTERANCE_S,
                min_silence_duration_ms=MAX_SILENCE_MS,
            )
            return [
                AudioChunk(audio=audio[ts["start"] : ts["end"]]) for ts in timestamps
            ]

    if __name__ == "__main__":
        seg = VADSegmenter()
        try:
            for chunk in seg.stream_mic():
                console.print(f"{chunk.uuid[:8]} [dim]{chunk.duration_s:.2f}s")
        except KeyboardInterrupt:
            console.print("[red]Stopped.[/red]")
