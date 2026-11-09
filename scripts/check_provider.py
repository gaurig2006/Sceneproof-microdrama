"""Verify the configured Gemini/OpenAI-compatible model without exposing secrets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sceneproof_ai.provider import OpenAICompatibleProvider, ProviderError


def main() -> int:
    provider = OpenAICompatibleProvider()
    if not provider.enabled:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Set GEMINI_API_KEY and SCENEPROOF_MODEL first.",
                }
            )
        )
        return 2
    try:
        print(json.dumps(provider.connection_check()))
        return 0
    except ProviderError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
