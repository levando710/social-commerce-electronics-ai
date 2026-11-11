from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from phone_assistant.conversation.catalog_answer import catalog_answer, answer_policy
from phone_assistant.evaluation.metrics.evaluate_handset_manual import read_ratings, score_rankings, answer_rating_summary
from phone_assistant.features.build_quarantined_features import quarantine_view

ROW = {"doc_id": "B012345678", "title": "iPhone 7", "catalog_facts": {"storage_gb": 128, "ram_gb": 128,
    "fact_warnings": ["ram_capacity_requires_review"]}}


class OfflineAnswerTests(unittest.TestCase):
    def test_catalog_summary_cites_source_and_does_not_confirm_suspect_ram(self):
        text = catalog_answer([ROW], {"preferences": ["gaming"]})
        self.assertIn("[B012345678]", text)
        self.assertIn("Bộ nhớ trong: 128 GB", text)
        self.assertNotIn("RAM: 128", text)
        self.assertIn("chưa được xác minh", text)
        self.assertIn("Chưa có phép đo hiệu năng", text)

    def test_offline_policy_cannot_be_overridden_by_request(self):
        with patch('phone_assistant.llm.gemini_client.generate_answer', side_effect=AssertionError("API must not run")) as generated:
            result = answer_policy("phone", [ROW], allow_llm=False, requested=True)
        generated.assert_not_called()
        self.assertEqual(result["mode"], "catalog_template")
        self.assertFalse(result["attempted"])

    def test_explicit_api_failure_falls_back_once_and_keeps_failure(self):
        with patch('phone_assistant.llm.gemini_client.generate_answer', return_value={"answer": None, "error": "quota", "enabled": True}) as generated:
            result = answer_policy("phone", [ROW], allow_llm=True)
        self.assertEqual(generated.call_count, 1)
        self.assertEqual(generated.call_args.kwargs["retries"], 0)
        self.assertEqual(result["mode"], "catalog_template_fallback")
        self.assertTrue(result["attempted"])
        self.assertEqual(result["error"], "quota")

    @unittest.skipUnless((ROOT / 'artifacts/cache/handset_runtime_v1/manifest.json').exists(), "Local handset runtime required")
    def test_default_handset_http_remains_offline_when_client_requests_llm(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        process = subprocess.Popen([sys.executable, "-B", str(ROOT / "src/run.py"), "chat_app", "--handsets", "--method", "legacy",
            "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if process.poll() is not None:
                    self.fail(process.stdout.read().decode("utf-8", errors="replace"))
                try:
                    if requests.get(url, timeout=1).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(.1)
            response = requests.post(url + "/api/query", json={"query": "Samsung phone 128GB", "use_llm": True,
                "use_query_llm": True}, timeout=20)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["answer_mode"], "catalog_template")
            self.assertFalse(body["llm_attempted"])
            self.assertFalse(body["llm_enabled"])
            self.assertTrue(body["answer"])
            self.assertIn(body["results"][0]["doc_id"], body["answer"])
        finally:
            process.terminate(); process.communicate(timeout=10)


class ManualRatingTests(unittest.TestCase):
    def test_unrated_or_uncertain_is_not_a_negative_vote(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            path.write_text("query_id,item_id,manual_relevance,reviewer\nq,a,,\nq,b,uncertain,\n", encoding="utf-8")
            ratings, unknown, count = read_ratings(path, {("q", "a"), ("q", "b")})
            self.assertEqual((ratings, unknown, count), ({}, 2, 2))
        rankings = {"a": [{"id": "q", "intent_id": "i", "ids": ["a"]}],
                    "b": [{"id": "q", "intent_id": "i", "ids": ["b"]}]}
        summary, common = score_rankings(rankings, {}, {"q": {"a", "b"}})
        self.assertEqual(common, [])
        self.assertIsNone(summary["a"]["metrics_on_common_queries"]["precision_strict@5"])

    def test_method_comparison_uses_common_fully_judged_queries(self):
        rankings = {"a": [{"id": "q", "intent_id": "i", "ids": ["x"]}],
                    "b": [{"id": "q", "intent_id": "i", "ids": ["y"]}]}
        pools = {"q": {"x", "y"}}
        summary, common = score_rankings(rankings, {("q", "x"): 2}, pools)
        self.assertEqual(common, [])
        summary, common = score_rankings(rankings, {("q", "x"): 2, ("q", "y"): 0}, pools)
        self.assertEqual(common, ["q"])
        self.assertEqual(summary["a"]["metrics_on_common_queries"]["precision_strict@5"], .2)
        self.assertEqual(summary["a"]["metrics_on_common_queries"]["pooled_ndcg@5"], 1.)
        self.assertEqual(summary["b"]["metrics_on_common_queries"]["pooled_ndcg@5"], 0.)

    def test_duplicate_or_outside_pool_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            path.write_text("query_id,item_id,manual_relevance,reviewer\nq,a,2,r\nq,a,0,r\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_ratings(path, {("q", "a")})
            with self.assertRaisesRegex(ValueError, "outside"):
                read_ratings(path, {("q", "b")})

    def test_missing_generated_answer_must_not_receive_manual_quality_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "answers.csv"
            path.write_text("query_id,raw_answer,manual_faithfulness,manual_relevance,manual_citation_correctness,reviewer\nq,,2,,,r\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed/missing"):
                answer_rating_summary(path)

    def test_edited_answer_cannot_be_scored_as_the_saved_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "answers.csv"
            path.write_text("query_id,raw_answer,manual_faithfulness,manual_relevance,manual_citation_correctness,reviewer\nq,fake,2,,,r\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from"):
                answer_rating_summary(path, {"q": {"answer": None}})

    def test_deleted_failure_rows_do_not_shrink_original_generation_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "answers.csv"
            path.write_text("query_id,raw_answer,manual_faithfulness,manual_relevance,manual_citation_correctness,reviewer\nq,real,2,2,2,r\n", encoding="utf-8")
            result = answer_rating_summary(path, {"q": {"answer": "real"}, "failed": {"answer": None}})
            self.assertEqual(result["requested_rows"], 2)
            self.assertEqual(result["provided_rows"], 1)
            self.assertEqual(result["unprovided_rows"], 1)
            self.assertEqual(result["generated_rows"], 1)

    def test_quarantine_changes_only_structured_ram_and_preserves_raw_input(self):
        products = [{"parent_asin": "p", "title": "Samsung phone", "details": {"RAM": "128 GB", "Memory Storage Capacity": "128 GB"},
                     "features": ["Original catalog description"], "images": []}]
        documents = [{"doc_id": "p", "title": "Samsung phone", "details": products[0]["details"]}]
        view, affected = quarantine_view(products, documents)
        self.assertNotIn("RAM", view[0]["details"])
        self.assertEqual(products[0]["details"]["RAM"], "128 GB")
        self.assertEqual(view[0]["features"], products[0]["features"])
        self.assertEqual(view[0]["details"]["Memory Storage Capacity"], "128 GB")
        self.assertEqual(affected[0]["removed_structured_fields"], {"RAM": "128 GB"})
