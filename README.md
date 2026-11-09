# SceneProof — Multimodal Adaptation QA

SceneProof reviews a localized microdrama screenplay against evidence extracted from the source video. It is a working hybrid AI prototype, not a newly trained foundation model.

**Author:** Gauri Gupta · Roll no. 230415 · Earth Sciences  
**Assignment:** Kuku AI Generalist — Problem 3: Find your own problem  
**GitHub Pages:** https://gaurig2006.github.io/sceneproof-microdrama-qa/

## Why this problem

After discussing the content workflow with a friend familiar with the industry, I investigated a narrow downstream risk: an Indian localization can change names and venues correctly while accidentally dropping the source hook, relationship conflict, or visual payoff. I checked this on a public Chinese microdrama episode and created a controlled Indian adaptation with five known defects.

I did not choose Problem 1 because credible promo ranking needs watch-time, CTR, or conversion labels that were unavailable for a valid three-hour evaluation. I did not choose Problem 2 because an honest legal assistant needs jurisdiction-aware retrieval, citations, privacy safeguards, and escalation—too high-stakes and broad for this prototype window.


The Pages site is a static verified demo. New-video Whisper/frame extraction runs through the Python backend locally or on a compute host.

## What runs for real

1. **Video/audio ingestion** — FFprobe validates the media and FFmpeg extracts mono 16 kHz audio.
2. **Speech AI** — `faster-whisper` detects Chinese and produces a timestamped English translation.
3. **Visual evidence** — scene-change frames plus uniform and ending frames are extracted with exact timestamps.
4. **OCR** — Tesseract reads visible on-screen text when the installed language pack supports it.
5. **Evidence alignment** — audio segments, OCR, captions and frame paths become one auditable evidence bundle.
6. **Adaptation model** — every run fits fresh word and character TF-IDF vectors over the source beats, adapted beats and supplied requirements. Timestamp alignment finds missing or semantically changed protected beats. Exact name and forbidden-term constraints are derived from the input map/rules.
7. **Human review** — the web UI keeps every proposed edit traceable and requires accept, reject or intentional-change decisions.

The Episode 1 validation run produces five issues, but no expected issue, show-specific name, venue, plot event, timestamp or issue count exists in the model code. The same tests include unseen names, forbidden terms and plot beats.

## Install

Python 3.11+ plus FFmpeg and Tesseract are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first Whisper run downloads the selected model. `small` is the default; `tiny` is faster for a quick laptop demo.

## Run the complete pipeline

The public-source episode video is intentionally not committed. Point `--video` at a copy you accessed legally.

```bash
python -m sceneproof_ai.cli \
  --video /path/to/episode-01.mp4 \
  --script dist/data/moonlight_is_not_late_ep01_flawed_indian_adaptation.txt \
  --character-map dist/data/moonlight_is_not_late_character_map.json \
  --rules dist/data/moonlight_is_not_late_adaptation_rules.txt \
  --ground-truth dist/data/moonlight_is_not_late_ep01_ground_truth.txt \
  --out runs
```

Each run creates unedited raw outputs:

- `asr_transcript.json`
- `frame_evidence.json` and timestamped JPEG frames
- `extracted_evidence.txt`
- `qa_report.json`
- `pipeline_result.json`

For a genuinely new episode, omit `--ground-truth`. The tool then reviews against automatically extracted evidence. Confidence is lower until a human verifies the source script.

## Run the web app with the real backend

```bash
uvicorn sceneproof_ai.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Upload a video, keep or replace the other three inputs, and click **Run multimodal AI**. The hosted static demo transparently falls back to the verified sample because static hosting cannot run Whisper.

## Optional model adapter

Set these variables to add vision captions and semantic QA without changing pipeline code:

```bash
export GEMINI_API_KEY=...                  # or SCENEPROOF_API_KEY / OPENAI_API_KEY
export SCENEPROOF_MODEL=<provider-model>   # selected at runtime, never embedded in QA logic
```

Gemini automatically uses its OpenAI-compatible endpoint when `GEMINI_API_KEY` is present. `SCENEPROOF_BASE_URL` may point to any other OpenAI-compatible provider. Without both a key and model name, Whisper plus the per-run alignment model still works; secrets are never shipped to the browser or source archive.

## Test

```bash
python -m unittest discover -s tests -v
node tests/model.test.js
```

## Honest limitations

- The default Tesseract installation may need the `chi_sim` language pack for Chinese subtitle OCR.
- Whisper speaker labels can be wrong; the evidence remains timestamped for correction.
- Without a vision model, the pipeline extracts frames but does not claim to understand visual actions automatically.
- TF-IDF alignment is explainable and runs locally, but it does not understand deep implication as well as a large language model. The optional configured LLM increases recall and also increases review risk, so every edit remains human-approved.
- This is an inference pipeline using pre-trained models. It is not a custom model trained on Kuku data.

