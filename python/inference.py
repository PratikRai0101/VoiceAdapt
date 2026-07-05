import time
from dataclasses import dataclass
from pathlib import Path
from tkinter.constants import N
from typing import Optional

from rich.console import Console
from torch.onnx.symbolic_opset11 import chunk
from torchaudio.functional.filtering import vad

console = Console()

DATA_DIR = Path.home() / ".voiceadapt"
MODELS_DIR = DATA_DIR / "models"
ADAPTERS_DIR = DATA_DIR / "adapters"

DEVICE = "mps"
COMPUTE_TYPE = "float32"
MODEL_SIZE = "small"


@dataclass
class TranscriptResult:
    text: str
    chunk_uuid: str
    language: str
    latency_ms: float
    adapter_ver: str = "base"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class InferenceEngine:
    def __init__(self, model_size=MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self._model:
            return
        from faster_whisper import WhisperModel

        console.print(
            f"[dim]Loading Whisper - {self.model_size} on {self.device}... [/dim]"
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=(MODELS_DIR),
        )
        ms = (time.perf_counter() - t0) * 1000
        console.print(
            f"[green] whisper - {self.model_size} loaded [/green] [dim] ({ms:.2f}ms) [/dim]"
        )

    def transcribe(self, chunk) -> TranscriptResult:
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            chunk.audio,
            language="en",
            beam_size=5,
            vad_filter=False,
            word_timestamp=False,
        )
        text = "".join(seg.text.strip() for seg in segments).strip()
        latency = (time.perf_counter() - t0) * 1000
        result = TranscriptResult(
            text=text,
            chunk_uuid=chunk.uuid,
            language=info.language,
            latency_ms=latency,
        )
        console.print(f"[bold]{text}[/bold] [dim]{latency:.0f}ms[/dim]")
        return result

    def reload_adapter(self, adapter_path=None):
        console.print(f"[cyan]Adapter reloaded: {adapter_path}[/cyan]")

    if __name__ == "__main__":
        import argparse

        import sounddevice as sd

        from python.vad import SAMPLE_RATE, AudioChunk

        parser = argparse.ArgumentParser()
        parser.add_argument("--mic", action="store_true")
        args = parser.parse_args()
        engine = InferenceEngine()
        if args.mic:
            console.print("[bold]Recording 5 seconds...[/bold]")
            audio = sd.rec(
                5 * SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            )
            sd.wait()
            chunk = AudioChunk(audio=audio[:, 0])
            result = engine.transcribe(chunk)
            console.print(f"[bold green] Result : [/bold green] {result.text}")
            console.print(f"[dim] Latency: {result.latency_ms:.0f}ms [/dim]")
