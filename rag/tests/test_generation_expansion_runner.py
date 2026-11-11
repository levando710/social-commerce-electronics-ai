import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.analysis.analyze_generation_expansion import analyze
from phone_assistant.evaluation.metrics import run_generation_expansion as runner
from phone_assistant.evaluation.reviews.prepare_generation_expansion import build_cases

PRICING = {"input_usd_per_million": 0.3, "output_usd_per_million": 2.5, "pricing_checked_date": "test"}
BUDGET = {"model": "gemini-2.5-flash", "scenario_usd": 0, **PRICING}
MANIFEST = {"cases_sha256": "test"}
JUDGMENT = json.dumps({"relevance": 2, "claims": [
    {"text": "phone", "supported": True, "evidence_type": "text", "doc_id": "product", "quote": "phone", "image_id": None}]})
USAGE = {"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 0}


class FakeProvider:
    """Stands in for call_gemini; records every outgoing request and can fail on a chosen call."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, parts, **settings):
        self.calls.append((parts, settings))
        if self.fail_on == len(self.calls):
            return {"enabled": True, "text": None, "error": "Gemini API 429: quota", "usage": None}
        is_judge = settings["response_mime_type"] == "application/json"
        return {"enabled": True, "text": JUDGMENT if is_judge else "phone [1]", "error": None, "usage": USAGE}


class GenerationExpansionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)
        image = self.root / "catalog.img"
        Image.new("RGB", (4, 4)).save(image, format="PNG")
        rows = [{"query_id": f"phone{n}_vi_0", "intent_id": f"phone{n}", "kind": "retrieval", "query": "phone",
                 "contexts": [{"doc_id": "product", "title": "phone", "product_images": [str(image)]}]} for n in range(2)]
        self.cases = build_cases({"development": rows})  # 2 intents x 3 questions x 2 arms
        self.output = self.root / "run"

    def run_with(self, provider, **overrides):
        settings = dict(budget_usd=5.0, max_requests=100, min_gap_seconds=0, caller=provider)
        settings.update(overrides)
        return runner.execute(self.cases, MANIFEST, BUDGET, self.output, **settings)

    def test_full_run_pairs_arms_journals_requests_and_reports_cost(self):
        provider = FakeProvider()
        status = self.run_with(provider)
        self.assertTrue(status["complete"])
        self.assertEqual(len(provider.calls), 2 * len(self.cases))
        self.assertEqual(status["requests_started"], len(provider.calls))
        for parts, settings in provider.calls:
            self.assertEqual(settings["temperature"], 0)
            self.assertEqual(settings["thinking_budget"], 0)
            self.assertEqual(settings["retries"], 0)
        records = [json.loads(line) for line in (self.output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in records:
            self.assertEqual(record["judgment"]["status"], "judged")
            self.assertIn("[product]", record["answer"])
        # 1000 prompt tokens and 100 output tokens per request at the test prices.
        self.assertAlmostEqual(status["estimated_spend_usd"], len(provider.calls) * (1000 * 0.3 + 100 * 2.5) / 1e6, places=6)
        self.assertTrue((self.output / "manual_answer_labels.csv").exists())
        labels = (self.output / "manual_answer_labels.csv").read_text(encoding="utf-8-sig").splitlines()[1]
        self.assertTrue(labels.endswith(",,,,,"))  # human_* columns stay blank

    def test_only_the_image_arm_carries_verified_images(self):
        provider = FakeProvider()
        self.run_with(provider)
        by_case = {}
        for event in runner.read_events(self.output / "requests.jsonl"):
            if event["event"] == "started":
                by_case[event["request_id"]] = None
        for (parts, _), request_id in zip(provider.calls, by_case):
            has_image = any("inline_data" in part for part in parts)
            if request_id.endswith("_text_only_generation") or request_id.endswith("_text_only_judge"):
                self.assertFalse(has_image, request_id)
            else:
                self.assertTrue(has_image, request_id)

    def test_rerun_of_a_finished_run_sends_nothing(self):
        self.run_with(FakeProvider())
        again = FakeProvider()
        self.run_with(again)
        self.assertEqual(again.calls, [])

    def test_request_cap_stops_before_exceeding_it(self):
        provider = FakeProvider()
        status = self.run_with(provider, max_requests=3)
        self.assertEqual(len(provider.calls), 3)
        self.assertFalse(status["complete"])
        self.assertFalse(status["stopped"])
        more = FakeProvider()
        self.run_with(more, max_requests=5)  # later session extends the cumulative cap and reuses saved responses
        self.assertEqual(len(more.calls), 2)

    def test_budget_refuses_a_request_that_could_exceed_it(self):
        provider = FakeProvider()
        status = self.run_with(provider, budget_usd=0.0001)
        self.assertEqual(provider.calls, [])
        self.assertEqual(status["requests_started"], 0)

    def test_first_api_error_stops_and_is_never_retried(self):
        provider = FakeProvider(fail_on=3)
        status = self.run_with(provider)
        self.assertEqual(len(provider.calls), 3)
        self.assertTrue(status["stopped"])
        self.assertTrue((self.output / "api_stop.json").exists())
        with self.assertRaisesRegex(RuntimeError, "resume-after-stop"):
            self.run_with(FakeProvider())
        after = FakeProvider()
        self.run_with(after, resume_after_stop=True)
        self.assertFalse((self.output / "api_stop.json").exists())
        self.assertTrue(list(self.output.glob("api_stop_cleared_*.json")))  # old marker preserved
        failed = [p for p in self.output.glob("request_*.json")
                  if json.loads(p.read_text(encoding="utf-8"))["status"] == "api_failed"]
        self.assertEqual(len(failed), 1)
        started = [e["request_id"] for e in runner.read_events(self.output / "requests.jsonl") if e["event"] == "started"]
        self.assertEqual(len(started), len(set(started)))  # every request id began exactly once

    def test_interrupted_request_is_charged_not_retried_and_halts(self):
        self.output.mkdir()
        request_id = f"{runner.request_order(self.cases)[0]['case_id']}_generation"
        runner.write_json(self.output / "plan.json", runner.make_plan(MANIFEST, BUDGET, self.cases))
        runner.append_jsonl(self.output / "requests.jsonl", {"event": "started", "request_id": request_id,
                            "started_unix": 1.0, "started_at_utc": "x", "worst_case_usd": 0.01, "output_tokens_cap": 1})
        provider = FakeProvider()
        with self.assertRaisesRegex(RuntimeError, "resume-after-stop"):
            self.run_with(provider)
        self.assertEqual(provider.calls, [])
        self.run_with(provider, resume_after_stop=True)
        called = [e["request_id"] for e in runner.read_events(self.output / "requests.jsonl")
                  if e["event"] == "started"]
        self.assertEqual(called.count(request_id), 1)
        saved = json.loads((self.output / f"request_{request_id}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "interrupted_request")
        ledger = runner.RequestLedger(self.output, runner.make_plan(MANIFEST, BUDGET, self.cases)["pricing"], 5, 100, 0)
        self.assertGreaterEqual(ledger.spent_usd(), 0.01)

    def test_changed_plan_is_rejected_for_an_existing_run(self):
        self.run_with(FakeProvider())
        changed = copy.deepcopy(BUDGET)
        changed["output_usd_per_million"] = 9
        with self.assertRaisesRegex(ValueError, "Plan changed"):
            runner.execute(self.cases, MANIFEST, changed, self.output, budget_usd=5, max_requests=10,
                           min_gap_seconds=0, caller=FakeProvider())

    def test_tampered_image_is_never_sent(self):
        Image.new("RGB", (4, 4), "red").save(Path(self.cases[1]["images"][0]["path"]), format="PNG")
        text_image = [c for c in self.cases if c["images"]][0]
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            runner.verified_image_parts(text_image["images"])

    def test_invalid_judge_output_is_recorded_without_stopping_the_run(self):
        class BadJudge(FakeProvider):
            def __call__(self, parts, **settings):
                response = super().__call__(parts, **settings)
                if settings["response_mime_type"]:
                    response["text"] = "not json"
                return response

        provider = BadJudge()
        status = self.run_with(provider)
        self.assertTrue(status["complete"])
        records = [json.loads(line) for line in (self.output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["judgment"]["status"] for r in records}, {"invalid_judgment"})

    def test_analysis_reports_missing_and_never_scores_empty_claims_as_one(self):
        self.run_with(FakeProvider())
        result = analyze(self.output, self.root / "analysis")
        self.assertEqual(result["coverage"]["not_recorded"], 0)
        self.assertEqual(result["human_labels_filled"], 0)
        self.assertEqual(result["overall"]["claim_micro"]["faithfulness"], 1.0)
        self.assertGreater(result["usage"]["billable_cost_usd_at_planned_prices"], 0)
        with self.assertRaises(FileExistsError):
            analyze(self.output, self.root / "analysis")


@unittest.skipUnless(runner.PREPARED.exists(), "prepared protocol assets are not present")
class PreparedProtocolPreflightTests(unittest.TestCase):
    def test_real_prepared_protocol_passes_preflight_without_network(self):
        with patch('phone_assistant.llm.gemini_client.requests.post') as post:
            cases, manifest, budget = runner.verify_prepared()
            post.assert_not_called()
        self.assertEqual(len(cases), manifest["planned_answers"])
        self.assertEqual(len(runner.request_order(cases)), len(cases))

    def test_tampered_cases_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            copy_path = Path(folder) / "prepared"
            shutil.copytree(runner.PREPARED, copy_path)
            with (copy_path / "cases.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "cases.jsonl"):
                runner.verify_prepared(copy_path)


if __name__ == "__main__":
    unittest.main()
