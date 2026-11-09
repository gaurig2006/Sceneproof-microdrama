from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from .models import FrameEvidence


class MediaError(RuntimeError):
    pass


def _run(command: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"Required command is missing: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()[-1200:]
        raise MediaError(f"{' '.join(command[:2])} failed: {detail}") from exc


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise MediaError(f"Install required media tools: {', '.join(missing)}")


def probe_video(video_path: Path) -> dict:
    require_media_tools()
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(video_path),
        ]
    )
    payload = json.loads(result.stdout)
    format_data = payload.get("format", {})
    video_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "filename": video_path.name,
        "duration_seconds": round(float(format_data.get("duration", 0)), 3),
        "size_bytes": int(format_data.get("size", video_path.stat().st_size)),
        "video_codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_rate": video_stream.get("r_frame_rate"),
        "audio_codec": audio_stream.get("codec_name"),
        "audio_sample_rate": audio_stream.get("sample_rate"),
        "audio_channels": audio_stream.get("channels"),
    }


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        timeout=300,
    )
    return wav_path


def detect_scene_changes(video_path: Path, threshold: float = 0.34) -> list[tuple[float, float]]:
    expression = f"select='gt(scene,{threshold})',metadata=print"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video_path),
            "-vf",
            expression,
            "-an",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        timeout=300,
    )
    if result.returncode != 0:
        return []
    timestamps = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    scores = [float(value) for value in re.findall(r"lavfi\.scene_score=([0-9.]+)", result.stderr)]
    return list(zip(timestamps, scores + [threshold] * max(0, len(timestamps) - len(scores))))


def choose_frame_times(
    duration: float,
    scene_changes: list[tuple[float, float]],
    interval: float = 6.0,
    max_frames: int = 24,
) -> list[tuple[float, str, float | None]]:
    candidates: dict[float, tuple[str, float | None]] = {}
    for second in range(0, max(1, math.ceil(duration)), max(1, round(interval))):
        candidates[round(min(float(second), max(duration - 0.08, 0.0)), 2)] = ("uniform", None)
    for timestamp, score in scene_changes:
        candidates[round(timestamp, 2)] = ("scene_change", round(score, 4))
    if duration > 0:
        candidates[round(max(duration - 0.12, 0.0), 2)] = ("ending", None)

    ordered = sorted((time, source, score) for time, (source, score) in candidates.items())
    if len(ordered) <= max_frames:
        return ordered

    fixed = [item for item in ordered if item[1] == "ending" or item[0] == 0]
    scene_slots = max(max_frames // 2, 1)
    strongest_scenes = sorted(
        (item for item in ordered if item[1] == "scene_change" and item not in fixed),
        key=lambda item: item[2] or 0,
        reverse=True,
    )[:scene_slots]
    must_keep = fixed + strongest_scenes
    remaining = [item for item in ordered if item not in must_keep]
    slots = max_frames - len(must_keep)
    if slots and remaining:
        step = max(len(remaining) / slots, 1)
        sampled = [remaining[min(int(i * step), len(remaining) - 1)] for i in range(slots)]
    else:
        sampled = []
    return sorted({item[0]: item for item in must_keep + sampled}.values())[:max_frames]


def extract_frames(video_path: Path, frame_dir: Path, selections: list[tuple[float, str, float | None]]) -> list[FrameEvidence]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[FrameEvidence] = []
    for index, (timestamp, source, score) in enumerate(selections, start=1):
        target = frame_dir / f"frame_{index:03d}_{timestamp:07.2f}s.jpg"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(720,iw)':-2",
                "-q:v",
                "3",
                str(target),
            ]
        )
        frames.append(
            FrameEvidence(
                timestamp=timestamp,
                image_path=str(target),
                source=source,
                scene_score=score,
            )
        )
    return frames


def available_tesseract_languages() -> set[str]:
    if not shutil.which("tesseract"):
        return set()
    result = subprocess.run(["tesseract", "--list-langs"], text=True, capture_output=True)
    lines = (result.stdout + result.stderr).splitlines()
    return {line.strip() for line in lines if re.fullmatch(r"[A-Za-z0-9_+-]+", line.strip())}


def ocr_frame(image_path: Path, language: str) -> tuple[str, float | None]:
    result = _run(
        ["tesseract", str(image_path), "stdout", "-l", language, "--psm", "6", "tsv"],
        timeout=60,
    )
    rows = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    words: list[str] = []
    confidences: list[float] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if text and confidence >= 20:
            words.append(text)
            confidences.append(confidence)
    return " ".join(words), round(sum(confidences) / len(confidences), 2) if confidences else None


def add_ocr(frames: list[FrameEvidence], preferred_language: str = "chi_sim+eng") -> tuple[list[FrameEvidence], str | None]:
    languages = available_tesseract_languages()
    if not languages:
        return frames, "Tesseract is unavailable, so on-screen text was not extracted."

    requested = [item for item in preferred_language.split("+") if item]
    primary = requested[0] if requested else "eng"
    if primary not in languages:
        return frames, f"OCR was skipped because the required Tesseract language pack '{primary}' is not installed."
    usable = [item for item in requested if item in languages]
    language = "+".join(usable)
    note = None if len(usable) == len(requested) else f"OCR used {language}; some requested language packs were unavailable."
    for frame in frames:
        frame.ocr_text, frame.ocr_confidence = ocr_frame(Path(frame.image_path), language)
    return frames, note

