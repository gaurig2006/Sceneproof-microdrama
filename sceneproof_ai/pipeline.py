from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .asr import transcribe_audio
from .media import (
    add_ocr,
    choose_frame_times,
    detect_scene_changes,
    extract_audio,
    extract_frames,
    probe_video,
)
from .models import FrameEvidence, PipelineResult, TranscriptSegment
from .provider import OpenAICompatibleProvider, ProviderError
from .qa import alignment_review, merge_model_issues


def format_timestamp(seconds: float) -> str:
    minutes, second = divmod(max(seconds, 0), 60)
    return f"{int(minutes):02d}:{second:05.2f}"


def build_evidence_text(segments: list[TranscriptSegment], frames: list[FrameEvidence]) -> str:
    lines = ["SOURCE EVIDENCE BUNDLE", "", "AUDIO TRANSCRIPT / TRANSLATION"]
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                "",
                f"**ASR SEGMENT {index:03d}:**",
                f"Start: ({format_timestamp(segment.start)}) | End: ({format_timestamp(segment.end)})",
                segment.text,
            ]
        )
    lines.extend(["", "VISUAL / OCR EVIDENCE"])
    for index, frame in enumerate(frames, start=1):
        observations = []
        if frame.caption:
            observations.append(frame.caption)
        if frame.ocr_text:
            observations.append(f"OCR: {frame.ocr_text}")
        if not observations:
            observations.append(f"Extracted {frame.source} frame for human verification")
        lines.extend(
            [
                "",
                f"**VISUAL FRAME {index:03d}:**",
                f"Start: ({format_timestamp(frame.timestamp)}) | End: ({format_timestamp(frame.timestamp + 0.10)})",
                " | ".join(observations),
            ]
        )
    return "\n".join(lines)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    *,
    video_path: Path,
    adapted_script: str,
    character_map: dict,
    rules: str,
    output_root: Path,
    ground_truth: str = "",
    asr_model: str = "small",
    frame_interval: float = 6.0,
    max_frames: int = 24,
    enable_ocr: bool = True,
    local_models_only: bool = False,
) -> PipelineResult:
    started = time.perf_counter()
    video_path = video_path.resolve()
    output_root = output_root.resolve()
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = output_root / run_id
    frame_dir = run_dir / "frames"
    run_dir.mkdir(parents=True, exist_ok=False)

    media = probe_video(video_path)
    wav_path = extract_audio(video_path, run_dir / "audio_16k_mono.wav")
    segments, transcript_meta = transcribe_audio(
        wav_path,
        model_name=asr_model,
        task="translate",
        local_files_only=local_models_only,
    )

    scene_changes = detect_scene_changes(video_path)
    selections = choose_frame_times(
        media["duration_seconds"],
        scene_changes,
        interval=frame_interval,
        max_frames=max_frames,
    )
    frames = extract_frames(video_path, frame_dir, selections)
    limitations: list[str] = []
    if enable_ocr:
        frames, ocr_note = add_ocr(frames)
        if ocr_note:
            limitations.append(ocr_note)
    else:
        limitations.append("OCR was disabled for this run.")

    provider = OpenAICompatibleProvider()
    mode = "local-asr-ocr-alignment-model"
    if provider.enabled:
        try:
            frames = provider.caption_frames(frames)
            mode = f"local-extraction+{provider.model}+alignment-model"
        except ProviderError as exc:
            limitations.append(f"Vision captions were skipped because the model API failed: {exc}")
    else:
        limitations.append("No vision-model key was provided; sampled frames are retained for human review but actions are not auto-captioned.")

    extracted_evidence = build_evidence_text(segments, frames)
    review_evidence = ground_truth.strip() or extracted_evidence
    if ground_truth.strip():
        limitations.append("QA used the supplied human-verified source script; ASR/OCR outputs remain available as raw audit evidence.")

    issues, alignment_metadata = alignment_review(adapted_script, character_map, rules, review_evidence)
    if provider.enabled:
        try:
            model_issues = provider.review_adaptation(review_evidence, adapted_script, character_map, rules)
            issues = merge_model_issues(issues, model_issues)
        except ProviderError as exc:
            limitations.append(f"Semantic model review was skipped because the model API failed: {exc}")

    transcript_payload = {**transcript_meta, "segments": [asdict(segment) for segment in segments]}
    elapsed = round(time.perf_counter() - started, 2)
    media["processing_seconds"] = elapsed
    media["scene_change_count"] = len(scene_changes)
    media["sampled_frame_count"] = len(frames)
    media["input_sha_note"] = "The original video is not copied into outputs; keep the licensed source separately."
    media["alignment_model"] = alignment_metadata

    result = PipelineResult(
        run_id=run_id,
        mode=mode,
        media=media,
        transcript=transcript_payload,
        frames=frames,
        issues=issues,
        limitations=limitations,
        output_directory=str(run_dir),
    )

    _write_json(run_dir / "asr_transcript.json", transcript_payload)
    _write_json(run_dir / "frame_evidence.json", [asdict(frame) for frame in frames])
    (run_dir / "extracted_evidence.txt").write_text(extracted_evidence, encoding="utf-8")
    _write_json(
        run_dir / "qa_report.json",
        {
            "run_id": run_id,
            "mode": mode,
            "issue_count": len(issues),
            "issues": [asdict(issue) for issue in issues],
            "limitations": limitations,
        },
    )
    _write_json(run_dir / "pipeline_result.json", result.to_dict())
    return result

