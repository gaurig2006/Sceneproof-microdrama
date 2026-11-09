from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .models import FrameEvidence


class ProviderError(RuntimeError):
    pass


def _json_from_text(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start_candidates = [index for index in (candidate.find("["), candidate.find("{")) if index >= 0]
    if not start_candidates:
        raise ProviderError("Model response did not contain JSON")
    start = min(start_candidates)
    end = max(candidate.rfind("]"), candidate.rfind("}"))
    return json.loads(candidate[start : end + 1])


class OpenAICompatibleProvider:
    """Small swappable VLM/LLM adapter; no SDK lock-in."""

    def __init__(self) -> None:
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.api_key = (
            os.getenv("SCENEPROOF_API_KEY", "").strip()
            or gemini_key
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        default_base = "https://generativelanguage.googleapis.com/v1beta/openai" if gemini_key else "https://api.openai.com/v1"
        self.base_url = (
            os.getenv("SCENEPROOF_BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or default_base
        ).rstrip("/")
        self.model = os.getenv("SCENEPROOF_MODEL", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def connection_check(self) -> dict[str, str | bool]:
        """Run a minimal authenticated request without printing or returning the key."""
        response = self._chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly SCENEPROOF_OK. This is an authentication "
                        "and model-availability check; do not add other text."
                    ),
                }
            ],
            max_tokens=20,
        )
        return {
            "ok": "SCENEPROOF_OK" in response.upper(),
            "model": self.model,
            "base_url": self.base_url,
        }

    def _chat(self, messages: list[dict], *, max_tokens: int = 1800) -> str:
        if not self.enabled:
            raise ProviderError("Set a provider API key and SCENEPROOF_MODEL")
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise ProviderError(f"Model API request failed: {exc}") from exc
        return payload["choices"][0]["message"]["content"]

    def caption_frames(self, frames: list[FrameEvidence]) -> list[FrameEvidence]:
        for offset in range(0, len(frames), 6):
            batch = frames[offset : offset + 6]
            content: list[dict] = [
                {
                    "type": "text",
                    "text": (
                        "You are extracting evidence from a vertical microdrama. Describe only visible actions, "
                        "character positions, emotions, objects, and shot changes. Do not infer dialogue. "
                        "Return a JSON array with timestamp and caption, one item per frame."
                    ),
                }
            ]
            for frame in batch:
                path = Path(frame.image_path)
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append({"type": "text", "text": f"Timestamp {frame.timestamp:.2f}s"})
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            response = self._chat([{"role": "user", "content": content}], max_tokens=1200)
            items = _json_from_text(response)
            for frame, item in zip(batch, items):
                frame.caption = str(item.get("caption", "")).strip()
        return frames

    def review_adaptation(self, evidence: str, script: str, character_map: dict, rules: str) -> list[dict]:
        prompt = f"""You are a conservative microdrama adaptation QA editor.
Compare SOURCE EVIDENCE against the INDIAN ADAPTATION using the CHARACTER MAP and RULES.
Return only a JSON array. Flag only material, evidence-backed problems. Each object must contain:
id, severity (critical|warning), category, title, explanation, evidence, timecode, proposal, confidence.
Do not invent a visual action or line. If evidence is insufficient, do not flag it.

SOURCE EVIDENCE:
{evidence[:18000]}

CHARACTER MAP:
{json.dumps(character_map, ensure_ascii=False)}

RULES:
{rules[:7000]}

INDIAN ADAPTATION:
{script[:18000]}
"""
        response = self._chat([{"role": "user", "content": prompt}], max_tokens=2600)
        payload = _json_from_text(response)
        return payload if isinstance(payload, list) else []

