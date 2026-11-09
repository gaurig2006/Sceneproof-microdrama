from pathlib import Path
import unittest

from sceneproof_ai.qa import AlignmentConfig


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (ROOT / "sceneproof_ai" / "qa.py", ROOT / "dist" / "model.js")


class HardcodingAuditTests(unittest.TestCase):
    def test_model_contains_no_fixture_specific_knowledge(self):
        forbidden = (
            "moonlight",
            "ktv",
            "huo mingxi",
            "meera",
            "aarav",
            "naina",
            "kavya",
            "rohan",
            "vikram",
            "drink throw",
            "cliffhanger",
            "expected issue",
            "issuecatalog",
        )
        for path in MODEL_FILES:
            source = path.read_text(encoding="utf-8").casefold()
            for term in forbidden:
                self.assertNotIn(term, source, f"{term!r} leaked into {path.name}")

    def test_alignment_thresholds_are_exposed_as_configuration(self):
        config = AlignmentConfig(
            relevance_floor=0.07,
            candidate_ratio=0.75,
            max_candidates_per_rule=3,
            minimum_time_overlap=0.20,
            preservation_floor=0.55,
            critical_confidence=0.80,
        )
        self.assertEqual(config.max_candidates_per_rule, 3)
        self.assertEqual(config.preservation_floor, 0.55)


if __name__ == "__main__":
    unittest.main()
