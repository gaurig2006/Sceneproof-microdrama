"""Refresh the checked-in QA report from the current generic alignment model."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sceneproof_ai.qa import alignment_review  # noqa: E402


DATA = ROOT / "dist" / "data"
OUTPUT = ROOT / "sample_outputs" / "episode_01"


def read(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


issues, metadata = alignment_review(
    read("moonlight_is_not_late_ep01_flawed_indian_adaptation.txt"),
    json.loads(read("moonlight_is_not_late_character_map.json")),
    read("moonlight_is_not_late_adaptation_rules.txt"),
    read("moonlight_is_not_late_ep01_ground_truth.txt"),
)
serialized = [asdict(issue) for issue in issues]

report_path = OUTPUT / "qa_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
report.update(
    {
        "mode": "local-asr-ocr-alignment-model",
        "issue_count": len(serialized),
        "issues": serialized,
        "alignment_model": metadata,
    }
)
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

result_path = OUTPUT / "pipeline_result.json"
result = json.loads(result_path.read_text(encoding="utf-8"))
result["mode"] = "local-asr-ocr-alignment-model"
result["issues"] = serialized
result.setdefault("media", {})["alignment_model"] = metadata
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Refreshed {len(serialized)} model-generated issues with {metadata['feature_count']} features")
