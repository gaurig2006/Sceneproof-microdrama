import json
import unittest
from pathlib import Path

from sceneproof_ai.qa import alignment_review


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dist" / "data"


def load(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


class AlignmentModelTests(unittest.TestCase):
    def test_sample_contains_exactly_five_planted_flaws(self):
        script = load("moonlight_is_not_late_ep01_flawed_indian_adaptation.txt")
        mapping = json.loads(load("moonlight_is_not_late_character_map.json"))
        rules = load("moonlight_is_not_late_adaptation_rules.txt")
        evidence = load("moonlight_is_not_late_ep01_ground_truth.txt")

        issues, metadata = alignment_review(script, mapping, rules, evidence)

        self.assertEqual(len(issues), 5)
        self.assertTrue(all(issue.timecode for issue in issues))
        self.assertFalse(metadata["hardcoded_show_knowledge"])
        self.assertGreater(metadata["feature_count"], 100)
        self.assertTrue(any(issue.detected_by.startswith("TF-IDF") for issue in issues))

    def test_character_map_check_is_data_driven(self):
        mapping = {
            "characters": [
                {"source_name": "李明", "romanized_name": "Li Ming", "adapted_name": "Arjun"}
            ]
        }
        issues, _ = alignment_review("Li Ming enters.", mapping, "Use approved names.", "")
        self.assertEqual(issues[0].id, "name-li-ming")
        self.assertIn("Arjun", issues[0].proposal)

    def test_unseen_forbidden_term_is_derived_from_rules(self):
        rules = '1. Location: Move the action to Mumbai. Do not retain the term "TeaHouse".'
        script = """SCENE 01
Scene start: 00:00.00
Scene end: 00:05.00

**ACTION:**
Start: (00:00.00) | End: (00:05.00)
They enter the TeaHouse.
"""
        issues, _ = alignment_review(script, {"characters": []}, rules, "")
        self.assertEqual(len(issues), 1)
        self.assertIn("teahouse", issues[0].id)

    def test_unseen_missing_beat_is_found_by_alignment(self):
        evidence = """**ACTION:**
Start: (00:10.00) | End: (00:15.00)
Riya tears the contract in front of Dev.
"""
        script = """**ACTION:**
Start: (00:00.00) | End: (00:05.00)
Riya enters the office.
"""
        rules = "1. Ending: Preserve Riya tearing the contract in front of Dev."
        issues, metadata = alignment_review(script, {"characters": []}, rules, evidence)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].category, "Missing required beat")
        self.assertGreater(metadata["feature_count"], 0)


if __name__ == "__main__":
    unittest.main()

