from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from .pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MAX_VIDEO_BYTES = 250 * 1024 * 1024

app = FastAPI(title="SceneProof AI", version="0.2.0")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "pipeline": "video → Whisper ASR → scene frames/OCR → evidence alignment → adaptation QA",
        "new_video_support": True,
    }


@app.post("/api/analyze")
async def analyze(
    source_video: UploadFile = File(...),
    adapted_script: str = Form(...),
    character_map: str = Form(...),
    rules: str = Form(...),
    ground_truth: str = Form(""),
) -> dict:
    try:
        parsed_map = json.loads(character_map)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="character_map must be valid JSON") from exc

    suffix = Path(source_video.filename or "source.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="sceneproof-upload-") as temporary:
        video_path = Path(temporary) / f"source{suffix}"
        with video_path.open("wb") as destination:
            shutil.copyfileobj(source_video.file, destination)
        if video_path.stat().st_size > MAX_VIDEO_BYTES:
            raise HTTPException(status_code=413, detail="Video exceeds the 250 MB prototype limit")
        try:
            result = await run_in_threadpool(
                run_pipeline,
                video_path=video_path,
                adapted_script=adapted_script,
                character_map=parsed_map,
                rules=rules,
                ground_truth=ground_truth,
                output_root=RUNS,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc
    return result.to_dict()


app.mount("/", StaticFiles(directory=ROOT / "dist", html=True), name="site")

