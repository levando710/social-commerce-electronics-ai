from collections import defaultdict
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.retrieval.search.bm25 import build_index
from phone_assistant.apps.chat_app import ChatServer
from phone_assistant.conversation.co_guiding import CoGuidingEngine
from phone_assistant.conversation.conversation_state import SessionStore
from phone_assistant.conversation.interaction_store import InteractionStore, LocalPreferenceService

DOCUMENTS = [
    {"doc_id": "a", "title": "Samsung Galaxy red phone", "text": "Samsung Galaxy red phone", "details": {}, "product_image_paths": []},
    {"doc_id": "b", "title": "Samsung Galaxy blue phone", "text": "Samsung Galaxy blue phone", "details": {}, "product_image_paths": []},
    {"doc_id": "c", "title": "Nokia basic keypad handset", "text": "Nokia basic keypad handset", "details": {}, "product_image_paths": []},
]


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "events.sqlite"
        self.store = InteractionStore(self.path, [d["doc_id"] for d in DOCUMENTS])
        self.visitor = self.store.create_visitor()

    def tearDown(self):
        self.temp.cleanup()

    def test_persistence_isolation_query_not_a_preference_and_idempotency(self):
        other = self.store.create_visitor()
        self.store.record(self.visitor, "query", query="gaming phones")
        self.assertEqual(self.store.profile(self.visitor)["item_weights"], {})
        self.store.record(self.visitor, "like", item_id="a", event_id="same-event")
        self.store.record(self.visitor, "like", item_id="a", event_id="same-event")
        with self.assertRaises(ValueError):
            self.store.record(self.visitor, "like", item_id="b", event_id="same-event")
        reopened = InteractionStore(self.path, [d["doc_id"] for d in DOCUMENTS])
        self.assertEqual(reopened.profile(self.visitor)["event_count"], 2)
        self.assertTrue(reopened.profile(self.visitor)["active_actions"]["a"]["like"])
        self.assertEqual(reopened.profile(other)["item_weights"], {})

    def test_reversible_actions_and_half_life(self):
        self.store.record(self.visitor, "like", item_id="a")
        self.store.record(self.visitor, "save", item_id="a")
        with self.store.connection() as connection:
            connection.execute("UPDATE events SET created=100")
        self.assertAlmostEqual(self.store.profile(self.visitor, now=100 + 30 * 86400)["item_weights"]["a"], 4.0)
        self.store.record(self.visitor, "unlike", item_id="a")
        self.store.record(self.visitor, "unsave", item_id="a")
        self.assertEqual(self.store.profile(self.visitor)["item_weights"], {})

    def test_preference_retrieves_similar_unseen_metadata_and_reset(self):
        service = LocalPreferenceService(DOCUMENTS, self.store)
        self.assertFalse(service.recommend(self.visitor)["personalized"])
        self.store.record(self.visitor, "like", item_id="a")
        feed = service.recommend(self.visitor)
        self.assertTrue(feed["personalized"])
        self.assertEqual(feed["results"][0]["doc_id"], "b")
        self.assertNotIn("a", {r["doc_id"] for r in feed["results"]})
        self.store.reset(self.visitor)
        self.assertFalse(service.recommend(self.visitor)["personalized"])

    def test_concurrent_retries_with_one_event_id_are_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            store = InteractionStore(Path(folder) / "history.sqlite3", {"a"})
            visitor = store.create_visitor()
            errors, barrier = [], threading.Barrier(6)

            def retry():
                barrier.wait()
                try:
                    store.record(visitor, "like", item_id="a", event_id="same-event")
                except Exception as error:
                    errors.append(error)

            workers = [threading.Thread(target=retry) for _ in range(6)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            self.assertEqual(errors, [])
            self.assertEqual(store.profile(visitor)["event_count"], 1)
            with self.assertRaisesRegex(ValueError, "different interaction"):
                store.record(visitor, "save", item_id="a", event_id="same-event")

    def test_invalid_events_are_rejected(self):
        for kwargs in ({"item_id": "outside"}, {"item_id": []}, {"item_id": "a", "event_id": 123}):
            with self.assertRaises(ValueError):
                self.store.record(self.visitor, "like", **kwargs)


class HistoryHTTPTests(unittest.TestCase):
    def test_cookie_profiles_co_guiding_session_binding_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            store = InteractionStore(Path(folder) / "events.sqlite", [d["doc_id"] for d in DOCUMENTS])
            preferences = LocalPreferenceService(DOCUMENTS, store)
            index = build_index(DOCUMENTS)
            class Recommender:
                data = SimpleNamespace(maps={"items": {d["doc_id"]: i for i, d in enumerate(DOCUMENTS)}}, train_seen=defaultdict(set))
                def recommend(self, user_index=None, top_k=5):
                    return {"personalized": False, "results": [{**d, "score": 0., "product_images": []} for d in DOCUMENTS][:top_k]}
            class Handler(ChatServer):
                documents = DOCUMENTS
                by_id = {d["doc_id"]: d for d in DOCUMENTS}
                interaction_store = store
                local_preferences = preferences
                sessions = SessionStore()
                co_guiding = CoGuidingEngine(DOCUMENTS, index, Recommender())
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with requests.Session() as first, requests.Session() as second:
                    response = first.get(url, timeout=5)
                    self.assertIn('data-history="1"', response.text)
                    self.assertIn("datn_visitor", first.cookies)
                    self.assertEqual(first.get(url + "/image", params={"path": str(Path(folder) / "events.sqlite")}, timeout=5).status_code, 404)
                    self.assertEqual(first.post(url + "/api/events", json={"kind": "like", "item_id": "a"}, timeout=5).status_code, 200)
                    feed = first.post(url + "/api/recommend", json={}, timeout=5).json()
                    self.assertEqual(feed["results"][0]["doc_id"], "b")
                    self.assertFalse(second.post(url + "/api/recommend", json={}, timeout=5).json()["personalized"])
                    query = first.post(url + "/api/query", json={"query": "Nokia", "use_llm": False}, timeout=5)
                    self.assertEqual(query.status_code, 200, query.text)
                    self.assertEqual([r["doc_id"] for r in query.json()["results"]], ["c"])
                    self.assertTrue(query.json()["feedback"]["personalized"])
                    sid = query.json()["session_id"]
                    self.assertEqual(second.post(url + "/api/query", json={"query": "Nokia", "session_id": sid, "use_llm": False}, timeout=5).status_code, 400)
                    self.assertEqual(first.post(url + "/api/query", json={"query": "Nokia", "user_index": 0, "use_llm": False}, timeout=5).status_code, 400)
                    self.assertEqual(first.post(url + "/api/profile/reset", json={}, timeout=5).status_code, 200)
                    self.assertEqual(first.get(url + "/api/profile", timeout=5).json()["profile"]["event_count"], 0)
                    self.assertEqual(first.post(url + "/api/query", json={"query": "Nokia", "session_id": sid, "use_llm": False}, timeout=5).status_code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
