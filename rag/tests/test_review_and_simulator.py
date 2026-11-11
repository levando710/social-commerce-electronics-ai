from pathlib import Path
import csv
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from http.server import ThreadingHTTPServer
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.apps.review_app import ReviewStore, ReviewHandler, Conflict, SCHEMAS
from phone_assistant.data.preparation.prepare_rule_conversations import examples
from phone_assistant.conversation.handset_reviews import select_reviews, ReviewEvidence
from phone_assistant.retrieval.search.bm25 import build_index
import hashlib


class ReviewStoreTests(unittest.TestCase):
    def make_store(self, root, kind="retrieval", rows=None):
        path = Path(root) / SCHEMAS[kind][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows is None:
            rows = [{"query_id": "q", "item_id": "p", "query": "phone", "title": "original", "manual_relevance": "", "reviewer": "", "notes": ""}]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return ReviewStore(root), path

    def test_loading_does_not_label_or_write_and_explicit_save_preserves_source_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            store, path = self.make_store(directory)
            before = path.read_bytes()
            task = store.task("retrieval", 0)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(task["row"]["manual_relevance"], "")
            result = store.save("retrieval", 0, {"manual_relevance": "2", "reviewer": "R1"}, task["version"])
            self.assertEqual(result["row"]["title"], "original")
            self.assertEqual(result["row"]["manual_relevance"], "2")
            self.assertEqual((Path(directory) / 'artifacts/evaluation/review_backups/retrieval.initial.csv').read_bytes(), before)
            self.assertTrue((Path(directory) / 'artifacts/evaluation/review_backups/changes.jsonl').exists())

    def test_stale_or_source_column_edits_and_missing_reviewer_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store, path = self.make_store(directory)
            task = store.task("retrieval", 0)
            with self.assertRaisesRegex(ValueError, "Only manual"):
                store.save("retrieval", 0, {"title": "changed"}, task["version"])
            with self.assertRaisesRegex(ValueError, "reviewer"):
                store.save("retrieval", 0, {"manual_relevance": "2"}, task["version"])
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(Conflict):
                store.save("retrieval", 0, {"manual_relevance": "0", "reviewer": "R"}, task["version"])

    def test_missing_answers_cannot_be_graded_and_ram_needs_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"query_id": "q", "raw_answer": "", "manual_faithfulness": "", "manual_relevance": "", "manual_citation_correctness": "", "reviewer": "", "notes": ""}
            store, _ = self.make_store(directory, "answers", [row])
            task = store.task("answers", 0)
            with self.assertRaisesRegex(ValueError, "Missing/failed"):
                store.save("answers", 0, {"manual_faithfulness": "2", "reviewer": "R"}, task["version"])
            store, _ = self.make_store(directory, "ram", [{"item_id": "p", "manual_correct_ram_gb": "", "reviewer": "", "notes": ""}])
            task = store.task("ram", 0)
            with self.assertRaisesRegex(ValueError, "source/evidence"):
                store.save("ram", 0, {"manual_correct_ram_gb": "8", "reviewer": "R"}, task["version"])

    def test_http_read_save_stale_and_origin_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            store, path = self.make_store(directory)
            handler = type("TestHandler", (ReviewHandler,), {"store": store, "documents": {}, "log_message": lambda *args: None})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                before = path.read_bytes()
                with urlopen(base) as response:
                    self.assertIn('Đánh giá', response.read().decode('utf-8'))
                with urlopen(base + "/api/task?kind=retrieval&index=0") as response:
                    task = json.load(response)
                self.assertEqual(path.read_bytes(), before)
                payload = json.dumps({"kind": "retrieval", "index": 0, "version": task["version"], "changes": {"manual_relevance": "2", "reviewer": "R1"}}).encode()
                def post(origin):
                    return urlopen(Request(base + "/api/save", data=payload, headers={"Content-Type": "application/json", "Origin": origin}))
                with self.assertRaises(HTTPError) as error:
                    post("https://external.example")
                self.assertEqual(error.exception.code, 403)
                with post(base) as response:
                    self.assertEqual(json.load(response)["row"]["manual_relevance"], "2")
                with self.assertRaises(HTTPError) as error:
                    post(base)
                self.assertEqual(error.exception.code, 409)
            finally:
                server.shutdown(); server.server_close(); thread.join()

    def test_answer_progress_requires_all_three_grades(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"query_id": "q", "raw_answer": "answer", "manual_faithfulness": "2", "manual_relevance": "", "manual_citation_correctness": "", "reviewer": "R", "notes": ""}
            store, _ = self.make_store(directory, "answers", [row])
            task = store.task("answers", 0)
            self.assertEqual(task["progress"]["decided"], 0)
            self.assertEqual(task["progress"]["partial"], 1)
            task = store.save("answers", 0, {"manual_relevance": "1", "manual_citation_correctness": "2"}, task["version"])
            self.assertEqual(task["progress"]["decided"], 1)


class SimulatorTests(unittest.TestCase):
    def test_public_reviews_match_exact_train_only_and_remain_opinions(self):
        uid = hashlib.sha256(b"author").hexdigest()
        data = SimpleNamespace(rows={"train": [{"user_id": uid, "item_id": "p", "timestamp": 1}],
            "validation": [], "test": [{"user_id": uid, "item_id": "q", "timestamp": 2}]})
        def review(item, time, text):
            return {"user_id": "author", "parent_asin": item, "timestamp": time, "text": text, "title": "Opinion"}
        source = [review("p", 1, "camera good"), review("p", 9, "future review"), review("q", 2, "target review")]
        rows = select_reviews(data, source + source)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["hardware_verified"])
        self.assertNotIn("user_id", rows[0])
        service = ReviewEvidence.__new__(ReviewEvidence)
        service.documents = {r["doc_id"]: r for r in rows}
        service.index = build_index(rows)
        self.assertEqual(service.search("p", "camera"), rows)
        self.assertEqual(service.search("q", "camera"), [])
        self.assertEqual(service.search("p", "battery"), [])

    def test_goals_excluded_from_profiles_and_query_users_do_not_cross_cohorts(self):
        def event(uid, item, timestamp):
            return {"user_id": uid, "item_id": item, "timestamp": timestamp, "rating": 5}
        train = [event("train", "p", 1), event("train", "q", 2), event("eval_a", "p", 1), event("eval_b", "p", 1)]
        validation = [event(uid, "q", 3) for uid in ("eval_a", "eval_b")]
        test = [event(uid, "r", 4) for uid in ("eval_a", "eval_b")]
        products = {item: {"details": {"Brand": "Samsung", "Memory Storage Capacity": "128 GB"},
            "title": f"DO NOT COPY {item}", "reviews": ["hidden target review"]} for item in ("p", "q", "r")}
        queries, labels, profiles, _ = examples(SimpleNamespace(rows={"train": train, "validation": validation, "test": test}), products)
        cohorts = {s: {r["user_id"] for r in labels if r["cohort"] == s} for s in ("train", "validation", "test")}
        self.assertEqual(cohorts["train"], {"train"})
        self.assertFalse(cohorts["validation"] & cohorts["test"])
        self.assertTrue(cohorts["validation"] and cohorts["test"])
        goals = {r["user_id"]: r for r in labels}
        for profile in profiles:
            self.assertNotIn(goals[profile["user_id"]]["item_id"], [r["item_id"] for r in profile["history"]])
        for query in queries:
            self.assertNotIn("DO NOT COPY", query["query"])
            self.assertNotIn("hidden target", query["query"])
            self.assertEqual(query["query"], " ".join(t["text"] for t in query["visible_turns"] if t["role"] == "user"))
