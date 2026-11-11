from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.audits.audit_generation_human import validate_review


class GenerationHumanTests(unittest.TestCase):
    def row(self, **changes):
        return {"case_id": "sample", "human_faithfulness": "", "human_relevance_0_1_2": "",
            "human_visual_support": "", "reviewer": "", "notes": "", **changes}

    def test_blank_and_partial_reviews_do_not_become_complete_scores(self):
        self.assertEqual(validate_review(self.row(), {"answer": None})["status"], "unreviewed")
        result = validate_review(self.row(human_relevance_0_1_2="2", reviewer="R1"), {"answer": "saved answer"})
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["faithfulness"])

    def test_failure_and_missing_reviewer_cannot_receive_a_score(self):
        with self.assertRaisesRegex(ValueError, "missing generation"):
            validate_review(self.row(human_relevance_0_1_2="2", reviewer="R1"), {"answer": None})
        with self.assertRaisesRegex(ValueError, "reviewer"):
            validate_review(self.row(human_relevance_0_1_2="2"), {"answer": "answer"})

    def test_nonfinite_out_of_range_or_unexplained_faithfulness_is_rejected(self):
        for value in ("nan", "inf", "-0.1", "1.1", "invalid", "0.5"):
            with self.assertRaises(ValueError):
                validate_review(self.row(human_faithfulness=value, reviewer="R1"), {"answer": "answer"})

    def test_valid_complete_review_keeps_human_identity_and_evidence_notes(self):
        result = validate_review(self.row(human_faithfulness="0.5", human_relevance_0_1_2="1",
            human_visual_support="uncertain", reviewer="R1", notes="Camera claim has no supplied image"), {"answer": "answer"})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["faithfulness"], .5)
        self.assertEqual(result["reviewer"], "R1")


if __name__ == "__main__":
    unittest.main()
