from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneProof multimodal adaptation QA")
    parser.add_argument("--video", required=True, help="Source microdrama video")
    parser.add_argument("--script", required=True, help="Adapted screenplay to review")
    parser.add_argument("--character-map", required=True, help="Chinese-to-target character map JSON")
    parser.add_argument("--rules", required=True, help="Adaptation rules text file")
    parser.add_argument("--ground-truth", help="Optional human-verified source screenplay")
    parser.add_argument("--out", default="runs", help="Directory for raw run outputs")
    parser.add_argument("--asr-model", default="small", help="faster-whisper model name")
    parser.add_argument("--frame-interval", type=float, default=6.0)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--local-models-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        video_path=Path(args.video),
        adapted_script=read_text(args.script),
        character_map=json.loads(read_text(args.character_map)),
        rules=read_text(args.rules),
        ground_truth=read_text(args.ground_truth) if args.ground_truth else "",
        output_root=Path(args.out),
        asr_model=args.asr_model,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        enable_ocr=not args.no_ocr,
        local_models_only=args.local_models_only,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

