import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phone_assistant.evaluation.metrics import evaluate_answer_relevance as module


def cases(count):
    return [{"case_id": f"c{i}", "pair_id": f"p{i // 2}", "arm": "text_only" if i % 2 == 0 else "text_image",
             "question_type": "q", "question": "Q?", "answer": "A."} for i in range(count)]


def good(parts, **kwargs):
    return {"text": json.dumps({"questions": ["a?", "b?", "c?"], "noncommittal": False}),
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 30}, "model": "m"}


class AnswerRelevanceTest(unittest.TestCase):
    def test_dry_run_sends_nothing_and_live_requires_caps(self):
        with tempfile.TemporaryDirectory() as folder:
            result = module.run(cases(4), Path(folder), False, None, None, call=lambda *a, **k: self.fail("sent"))
            self.assertTrue(result["dry_run"])
            with self.assertRaises(SystemExit):
                module.run(cases(4), Path(folder), True, None, None)

    def test_stops_at_first_error_and_never_retries_the_failed_case(self):
        sent = []
        def flaky(parts, **kwargs):
            sent.append(1)
            return {"text": None, "error": "boom"} if len(sent) == 2 else good(parts)
        with tempfile.TemporaryDirectory() as folder:
            first = module.run(cases(5), Path(folder), True, 1.0, 10, call=flaky, sleep=lambda s: None)
            self.assertEqual((first["sent"], len(sent)), (2, 2))
            self.assertIn("first error", first["stopped"])
            again = module.run(cases(5), Path(folder), True, 1.0, 10, call=good, sleep=lambda s: None)
            self.assertEqual(again["sent"], 3)  # c0 and the failed c1 are not sent again

    def test_request_cap_and_budget_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(module.run(cases(5), Path(folder), True, 1.0, 2, call=good, sleep=lambda s: None)["sent"], 2)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(module.run(cases(5), Path(folder), True, 0.0001, 10, call=good, sleep=lambda s: None)["sent"], 0)

    def test_parse_questions(self):
        questions, noncommittal = module.parse_questions('{"questions": ["x?", " ", "y?"], "noncommittal": true}')
        self.assertEqual((questions, noncommittal), (["x?", "y?"], True))
        with self.assertRaises(ValueError):
            module.parse_questions('{"questions": []}')


if __name__ == "__main__":
    unittest.main()
