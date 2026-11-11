import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.metrics import evaluate_spec_extraction as spec

JSON_ANSWER = json.dumps({
    "color": {"value": "Black", "source": "own_text", "quote": "Black"},
    "brand": {"value": "Acme", "source": "own_text", "quote": "Acme"},
    "screen_inch": None, "os": None, "storage_gb": None, "ram_gb": None})


def product(asin, **details):
    return {"parent_asin": asin, "title": f"Acme Gadget {asin} Black 15.6 inch laptop 8GB RAM", "store": "Acme",
            "features": ["Runs Windows 11"], "description": ["A solid laptop."], "categories": ["Electronics"],
            "details": details}


class GoldAndEvidenceTests(unittest.TestCase):
    def test_gold_is_normalized_from_structured_fields(self):
        gold = spec.gold_attributes(product("a", Color="Black", Brand="Acme", **{"Screen Size": "15.6 Inches",
                                    "Memory Storage Capacity": "1 TB", "RAM": "8 GB", "Operating System": "Windows 11"}))
        self.assertEqual(gold, {"color": "Black", "brand": "Acme", "screen_inch": 15.6, "storage_gb": 1024.0,
                                "ram_gb": 8.0, "os": "Windows 11"})

    def test_model_input_never_contains_the_details_fields_or_their_derived_tokens(self):
        item = product("a", **{"Item model number": "SECRET-MODEL-1", "RAM": "64 GB"})
        item["description"] = ["plain text"]
        text = spec.evidence_text(item)
        self.assertNotIn("SECRET-MODEL-1", text)
        self.assertNotIn("ram=", text.lower())
        parts, own, _ = spec.build_request(item, [], False, False, None, None)
        self.assertNotIn("SECRET-MODEL-1", parts[0]["text"])
        self.assertEqual(len(parts), 1)  # no image part unless requested

    def test_selection_is_deterministic_disjoint_and_respects_quotas(self):
        products = [product(f"P{n:03d}", Color="Black") for n in range(300)]
        first = spec.select_items(products, quotas={"color": 50}, seed=1)
        self.assertEqual(first, spec.select_items(products, quotas={"color": 50}, seed=1))
        self.assertFalse(set(first["dev"]) & set(first["test"]))
        self.assertEqual(len(first["dev"]) + len(first["test"]), 50)
        self.assertNotEqual(first, spec.select_items(products, quotas={"color": 50}, seed=2))


class MatchingTests(unittest.TestCase):
    def test_numeric_and_categorical_matching_rules(self):
        self.assertTrue(spec.matches("ram_gb", 8.0, "8"))
        self.assertFalse(spec.matches("ram_gb", 8.0, 16))
        self.assertTrue(spec.matches("screen_inch", 15.6, 15.6))
        self.assertFalse(spec.matches("screen_inch", 15.6, 14))
        self.assertTrue(spec.matches("color", "Black/Silver", "black"))
        self.assertFalse(spec.matches("color", "Black", "white"))
        self.assertTrue(spec.matches("brand", "SAMSUNG Electronics", "Samsung"))
        self.assertFalse(spec.matches("brand", "Samsung", "LG"))
        self.assertTrue(spec.matches("storage_gb", 1024.0, 1000))  # 1 TB read with the decimal convention
        self.assertTrue(spec.matches("storage_gb", 700 / 1024, 0.7))  # 700 MB
        self.assertFalse(spec.matches("storage_gb", 32.0, 4))
        self.assertTrue(spec.matches("color", "Silver", "Aluminum Metallic"))
        self.assertFalse(spec.matches("color", "Silver", "Black"))
        self.assertTrue(spec.matches("os", "Windows 10 Home", "Windows 11"))
        self.assertFalse(spec.matches("os", "Android", "iOS"))

    def test_gold_in_text_distinguishes_stated_from_missing_values(self):
        self.assertTrue(spec.gold_in_text("ram_gb", 8.0, "laptop with 8GB memory"))
        self.assertFalse(spec.gold_in_text("ram_gb", 16.0, "laptop with 8GB memory"))
        self.assertFalse(spec.gold_in_text("color", "Black", "a silver laptop"))


class GoldHygieneTests(unittest.TestCase):
    def test_non_color_gold_is_not_scored(self):
        self.assertFalse(spec.valid_gold("color", "4 channel 960H DVR Kit"))
        self.assertTrue(spec.valid_gold("color", "Black/Silver"))
        item = product("a", Color="4 channel 960H DVR Kit")
        record = {"asin": "a", "condition": "text", "prediction": {"color": {"value": "black", "source": "own_text", "grounded": True}}}
        self.assertEqual(spec.score_records([record], [item]), {})


class ParsingAndScoringTests(unittest.TestCase):
    def test_quote_must_come_from_the_evidence_it_cites(self):
        answer = json.dumps({"color": {"value": "red", "source": "own_text", "quote": "crimson finish"},
                             "brand": {"value": "Acme", "source": "neighbor", "quote": "Acme neighbor text"},
                             "screen_inch": None, "os": None, "storage_gb": None, "ram_gb": None})
        parsed = spec.parse_prediction(answer, "A black laptop", {"N1": "Acme neighbor text here"})
        self.assertFalse(parsed["color"]["grounded"])
        self.assertTrue(parsed["brand"]["grounded"])
        with self.assertRaises(ValueError):
            spec.parse_prediction("[1, 2]", "", {})

    def test_outcomes_and_rates_count_correct_wrong_and_absent(self):
        items = [product("a", Color="Black", Brand="Acme"), product("b", Color="Red", Brand="Acme")]
        records = [
            {"asin": "a", "condition": "text", "prediction": {"color": {"value": "black", "source": "own_text", "grounded": True}, "brand": None}},
            {"asin": "b", "condition": "text", "prediction": {"color": {"value": "blue", "source": "own_text", "grounded": True}, "brand": None}}]
        summary = spec.summarize(spec.score_records(records, items))
        color = summary["color"]["text"]
        self.assertEqual((color["correct"], color["wrong"], color["absent"]), (1, 1, 0))
        self.assertEqual(color["precision"], 0.5)
        self.assertEqual(summary["brand"]["text"]["absent"], 2)
        self.assertEqual(summary["brand"]["text"]["recall"], 0.0)

    def test_rule_baseline_reads_listing_text_only(self):
        found = spec.rule_baseline(product("a", RAM="64 GB"))
        self.assertEqual(found["ram_gb"]["value"], 8.0)  # from the title, not from the hidden 64 GB gold
        self.assertEqual(found["screen_inch"]["value"], 15.6)
        self.assertEqual(found["os"]["value"], "windows")


class PairedDifferenceTests(unittest.TestCase):
    def test_paired_delta_is_second_minus_first_on_shared_items(self):
        cells = {("color", "text"): [{"asin": str(i), "outcome": "absent"} for i in range(20)],
                 ("color", "text_image"): [{"asin": str(i), "outcome": "correct" if i < 15 else "wrong"} for i in range(20)]}
        result = spec.paired_differences(cells, repetitions=200)["color"]["text_image_minus_text"]
        self.assertEqual(result["items"], 20)
        self.assertAlmostEqual(result["recall_delta"]["mean"], 0.75)
        self.assertAlmostEqual(result["wrong_rate_delta"]["mean"], 0.25)
        self.assertGreater(result["recall_delta"]["ci95"][0], 0)


class RunTests(unittest.TestCase):
    def test_run_journals_scores_and_does_not_repeat_finished_work(self):
        calls = []

        def fake(parts, **settings):
            calls.append(parts)
            return {"enabled": True, "text": JSON_ANSWER, "error": None,
                    "usage": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 0}}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            products = [product(f"P{n:02d}", Color="Black", Brand="Acme") for n in range(30)]
            for item in products:
                Image.new("RGB", (4, 4)).save(images / f"{item['parent_asin']}.img", format="PNG")
                path = images / f"{item['parent_asin']}.img"
                (images / f"{item['parent_asin']}.json").write_text(json.dumps(
                    {"status": "ok", "sha256": spec.sha256(path)}), encoding="utf-8")
            with patch.object(spec, "IMAGE_DIRECTORY", images):
                result = spec.run("test", root / "out", products, budget_usd=5, max_requests=1000, live=True,
                                  caller=fake, min_gap_seconds=0)
                self.assertFalse(result["stopped"])
                first_calls = len(calls)
                self.assertGreater(first_calls, 0)
                with_image = [c for c in calls if len(c) > 1]
                self.assertTrue(with_image)  # image conditions send a picture
                spec.run("test", root / "out", products, budget_usd=5, max_requests=1000, live=True,
                         caller=fake, min_gap_seconds=0)
                self.assertEqual(len(calls), first_calls)  # finished work is not repeated
                summary = spec.analyze(root / "out", products, "test")
            self.assertIn("rules", summary["summary"]["color"])
            self.assertEqual(summary["summary"]["color"]["text"]["wrong"], 0)

    def test_first_api_failure_stops_the_run_without_retrying(self):
        calls = []

        def failing(parts, **settings):
            calls.append(1)
            return {"enabled": True, "text": None, "error": "OpenAI API 429: quota", "usage": None}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            products = [product(f"P{n:02d}", Color="Black") for n in range(30)]
            result = spec.run("test", root / "out", products, budget_usd=5, max_requests=1000, live=True,
                              caller=failing, min_gap_seconds=0)
            self.assertTrue(result["stopped"])
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(RuntimeError, "resume-after-stop"):
                spec.run("test", root / "out", products, budget_usd=5, max_requests=1000, live=True,
                         caller=failing, min_gap_seconds=0)
            ok_calls = []

            def working(parts, **settings):
                ok_calls.append(1)
                return {"enabled": True, "text": JSON_ANSWER, "error": None, "usage": {"promptTokenCount": 10, "candidatesTokenCount": 5}}

            images = root / "images"
            images.mkdir()
            for item in products:
                path = images / f"{item['parent_asin']}.img"
                Image.new("RGB", (4, 4)).save(path, format="PNG")
                (images / f"{item['parent_asin']}.json").write_text(json.dumps(
                    {"status": "ok", "sha256": spec.sha256(path)}), encoding="utf-8")
            with patch.object(spec, "IMAGE_DIRECTORY", images):
                spec.run("test", root / "out", products, budget_usd=5, max_requests=1000, live=True, caller=working,
                         min_gap_seconds=0, resume_after_stop=True)
            self.assertGreater(len(ok_calls), 0)
            started = [e["request_id"] for e in spec.read_jsonl(root / "out" / "requests.jsonl") if e["event"] == "started"]
            self.assertEqual(len(started), len(set(started)))  # the failed request was not started a second time


if __name__ == "__main__":
    unittest.main()
