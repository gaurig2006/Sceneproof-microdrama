import io
import os
import unittest
from unittest.mock import patch

from sceneproof_ai.provider import OpenAICompatibleProvider


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class ProviderAdapterTests(unittest.TestCase):
    def test_gemini_adapter_uses_runtime_configuration(self):
        payload = b'{"choices":[{"message":{"content":"SCENEPROOF_OK"}}]}'
        env = {
            "GEMINI_API_KEY": "test-only-secret",
            "SCENEPROOF_MODEL": "runtime-selected-model",
        }
        with patch.dict(os.environ, env, clear=True):
            provider = OpenAICompatibleProvider()
            with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)) as request:
                result = provider.connection_check()

        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "runtime-selected-model")
        sent_request = request.call_args.args[0]
        self.assertTrue(sent_request.full_url.endswith("/chat/completions"))
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer test-only-secret")


if __name__ == "__main__":
    unittest.main()
