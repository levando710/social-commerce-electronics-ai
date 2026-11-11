from pathlib import Path
import json
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from phone_assistant.retrieval.serving.connected_serving import ConnectedRuntime, ConversationalQueryRanking, ConnectedDenseRetriever
from phone_assistant.common.image_mime import image_mime
from phone_assistant.llm.gemini_client import image_part


class ConnectedServingTests(unittest.TestCase):
    def test_downloaded_img_uses_content_mime(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "download.img"
            for format, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
                with Image.new("RGB", (8, 8)) as image:
                    image.save(path, format=format)
                self.assertEqual(image_mime(path), mime)
                self.assertEqual(image_part(path)["inline_data"]["mime_type"], mime)

    def test_long_query_serving_pools_chunks_and_uses_saved_projection(self):
        class Encoder:
            def encode(self, chunks, **kwargs):
                self.chunks = chunks
                return np.asarray([[1., 0.] if i % 2 == 0 else [0., 1.] for i in range(len(chunks))])
        encoder = Encoder()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "joint" / "seed_42"
            model.mkdir(parents=True)
            np.savez(model / "embeddings.npz", query_weight=np.eye(2), query_bias=np.zeros(2))
            report = directory / "experiment.json"
            report.write_text(json.dumps({"feature_manifest": {"text_encoder": {"model": "unused", "revision": "unused"}}}))
            service = SimpleNamespace(method="joint", seed=42, items=np.eye(2), data=SimpleNamespace(item_ids=["a", "b"]))
            dense = SimpleNamespace(lock=__import__("threading").RLock(), encoders={"text": encoder})
            ranking = ConversationalQueryRanking(service, directory, report, dense=dense)
            result = ranking.rank(" ".join(["word"] * 100))
            self.assertEqual(len(encoder.chunks), 2)
            self.assertAlmostEqual(result[0][1], 2 ** -.5)
            self.assertAlmostEqual(result[1][1], 2 ** -.5)

    @unittest.skipUnless((ROOT / 'artifacts/cache/connected_runtime_v1/manifest.json').exists(), "Local connected assets required")
    def test_verified_runtime_loads_without_loading_encoders(self):
        runtime = ConnectedRuntime()
        service = runtime.recommender()
        dense = ConnectedDenseRetriever(runtime.dense, list(service.documents.values()))
        self.assertEqual(service.data.n_items, 4435)
        self.assertEqual(dense.encoders, {})
        self.assertEqual(len(service.recommend(0, 10)["results"]), 10)

    @unittest.skipUnless((ROOT / 'artifacts/cache/connected_runtime_v1/manifest.json').exists(), "Local connected assets required")
    def test_connected_http_query_image_and_browser_history_without_llm(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, "-B", str(ROOT / "src/run.py"), "chat_app", "--connected", "--historical-mixed-catalog", "--method", "legacy",
                "--port", str(port), "--interaction-db", str(Path(temporary) / "profile.sqlite")]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                client = requests.Session()
                url = f"http://127.0.0.1:{port}"
                for _ in range(80):
                    if process.poll() is not None:
                        self.fail(process.stdout.read().decode("utf-8", errors="replace"))
                    try:
                        if client.get(url, timeout=1).status_code == 200:
                            break
                    except requests.ConnectionError:
                        pass
                    time.sleep(.1)
                response = client.post(url + "/api/query", json={"query": "case", "use_llm": False}, timeout=30)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertFalse(result["llm_enabled"])
                self.assertTrue(result["results"])
                self.assertEqual(result["query_plan"]["constraints"]["product_type"], "case")
                for row in result["results"]:
                    self.assertTrue(row["constraint_check"]["eligible"])
                    self.assertIn("case", row["catalog_facts"]["product_types"])
                row = result["results"][0]
                image = client.get(url + "/image", params={"path": row["product_images"][0]}, timeout=5)
                self.assertEqual(image.status_code, 200)
                self.assertTrue(image.headers["Content-Type"].startswith("image/"))
                self.assertEqual(client.get(url + "/image", params={"path": str(ROOT / "src/run.py")}).status_code, 404)
                event = client.post(url + "/api/events", json={"kind": "save", "item_id": row["doc_id"], "event_id": "test-save"})
                self.assertEqual(event.status_code, 200, event.text)
                self.assertTrue(event.json()["profile"]["active_actions"][row["doc_id"]]["save"])
                followup = client.post(url + "/api/query", json={"query": "for iPhone 6", "use_llm": False,
                    "session_id": result["session_id"]}, timeout=30)
                self.assertEqual(followup.status_code, 200, followup.text)
                self.assertEqual(followup.json()["query_plan"]["constraints"]["product_type"], "case")
                self.assertEqual(client.post(url + "/api/profile/reset", json={}).status_code, 200)
                expired = client.post(url + "/api/query", json={"query": "case", "use_llm": False,
                    "session_id": result["session_id"]}, timeout=10)
                self.assertEqual(expired.status_code, 400)
            finally:
                process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
