from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class FrameEvidence:
    timestamp: float
    image_path: str
    source: str
    scene_score: float | None = None
    ocr_text: str = ""
    ocr_confidence: float | None = None
    caption: str = ""


@dataclass(slots=True)
class ReviewIssue:
    id: str
    severity: str
    category: str
    title: str
    explanation: str
    evidence: str
    timecode: str
    proposal: str
    confidence: float
    detected_by: str


@dataclass(slots=True)
class PipelineResult:
    run_id: str
    mode: str
    media: dict[str, Any]
    transcript: dict[str, Any]
    frames: list[FrameEvidence]
    issues: list[ReviewIssue]
    limitations: list[str] = field(default_factory=list)
    output_directory: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_directory"] = str(Path(self.output_directory))
        return payload

