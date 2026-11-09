from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .models import TranscriptSegment


class ASRError(RuntimeError):
    pass


def read_pcm16_wav(path: Path) -> np.ndarray:
    """Decode our FFmpeg-produced WAV without relying on PyAV."""
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
            raise ASRError("ASR expects mono 16 kHz PCM16 WAV audio")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def transcribe_audio(
    wav_path: Path,
    model_name: str = "small",
    *,
    task: str = "translate",
    local_files_only: bool = False,
) -> tuple[list[TranscriptSegment], dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ASRError("Install faster-whisper to run speech recognition") from exc

    try:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only,
        )
    except Exception as exc:
        hint = " The model is not cached; allow the first download or choose an installed model." if local_files_only else ""
        raise ASRError(f"Could not load Whisper model '{model_name}'.{hint}") from exc

    audio = read_pcm16_wav(wav_path)
    segment_iterator, info = model.transcribe(
        audio,
        task=task,
        beam_size=3,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    segments = [
        TranscriptSegment(start=round(item.start, 2), end=round(item.end, 2), text=item.text.strip())
        for item in segment_iterator
        if item.text.strip()
    ]
    metadata = {
        "engine": "faster-whisper",
        "model": model_name,
        "task": task,
        "detected_language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "segment_count": len(segments),
    }
    return segments, metadata

