from pathlib import Path
import socket
import json
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from phone_assistant.data.phone_scope import handset_decision, handset_documents, HandsetRanking, HandsetRecommender


class PhoneScopeTests(unittest.TestCase):
    def test_handsets_include_iphone_and_feature_phone_but_not_compatibility_mentions(self):
        for title in ("Apple iPhone 13 128GB unlocked", "Nokia 1661 Gsm Unlocked Cell Phone", "Samsung Galaxy S21 smartphone"):
            self.assertTrue(handset_decision({"title": title})[0], title)
        for title in ("case for iPhone 13", "iPhone LCD screen replacement", "Phone signal booster", "Tools for iPhone",
                      "Samsung tablet", "Phone charm strap", "iPhone juice pack Air"):
            self.assertFalse(handset_decision({"title": title})[0], title)
        self.assertFalse(handset_decision({"title": "Wireless device", "categories": ["Cell Phones & Accessories"]})[0])
        self.assertFalse(handset_decision({"title": "mophie for iPhone", "categories": ["Basic Cases"]})[0])
        self.assertFalse(handset_decision({"title": "PRIZM NOKIA 8200 SERIES CLEAR.", "details": {"Manufacturer": "Nokia"}})[0])

    def test_scope_is_applied_before_dense_candidate_limit(self):
        class Ranking:
            def search(self, query, *, channel, top_k):
                return [(str(i), 200 - i) for i in range(top_k)]
            def rank(self, query, top_k):
                return self.search(query, channel="text", top_k=top_k)
        scoped = HandsetRanking(Ranking(), {"101", "102"}, 150)
        self.assertEqual(scoped.search("phone", top_k=1), [("101", 99)])
        self.assertEqual(scoped.rank("phone", top_k=1), [("101", 99)])

    def test_recommendations_rank_allowed_items_before_limiting_and_keep_seen_excluded(self):
        data = SimpleNamespace(item_ids=["accessory", "phone", "seen"], item_degree=np.asarray([100, 1, 2]),
            train_seen=[{2}])
        base = SimpleNamespace(data=data, users=np.asarray([[1.]]), items=np.asarray([[100.], [1.], [2.]]),
            bias=np.zeros(3), recommend=lambda user, k: {"results": [{"doc_id": "accessory"}]})
        scoped = HandsetRecommender(base, [{"doc_id": "phone"}, {"doc_id": "seen"}])
        self.assertEqual(scoped.recommend(0, 1)["results"][0]["doc_id"], "phone")
        self.assertEqual(scoped.recommend(None, 1)["results"][0]["doc_id"], "seen")

    @unittest.skipUnless((ROOT / 'artifacts/cache/connected_runtime_v1/manifest.json').exists(), "Local connected assets required")
    def test_default_connected_http_excludes_accessories_from_query_and_feed(self):
        documents = [json.loads(line) for line in (ROOT / 'artifacts/cache/connected_runtime_v1/documents.jsonl').read_text(encoding="utf-8").splitlines()]
        allowed = {doc["doc_id"] for doc in handset_documents(documents)}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen([sys.executable, "-B", str(ROOT / "src/run.py"), "chat_app", "--connected", "--method", "legacy",
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
            response = requests.post(url + "/api/query", json={"query": "phone", "use_llm": False}, timeout=30)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["results"])
            for row in result["results"] + result["recommendations"]:
                self.assertIn(row["doc_id"], allowed)
            accessory = requests.post(url + "/api/query", json={"query": "case for iPhone 13", "use_llm": False}, timeout=30)
            self.assertEqual(accessory.status_code, 200, accessory.text)
            self.assertEqual(accessory.json()["results"], [])
            feed = requests.post(url + "/api/recommend", json={"top_k": 100}, timeout=10)
            self.assertEqual(feed.status_code, 200, feed.text)
            self.assertTrue(feed.json()["results"])
            self.assertTrue(all(row["doc_id"] in allowed for row in feed.json()["results"]))
        finally:
            process.terminate()
            process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
