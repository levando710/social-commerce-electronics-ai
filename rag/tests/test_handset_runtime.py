from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.data.phone_scope import handset_decision


@unittest.skipUnless((ROOT / 'artifacts/cache/handset_runtime_v1/manifest.json').exists(), "Handset-only runtime assets required")
class HandsetRuntimeTests(unittest.TestCase):
    def test_loaded_model_really_uses_new_phone_only_split(self):
        runtime = HandsetRuntime()
        service = runtime.recommender("lightgcn", 42)
        self.assertEqual(service.data.n_items, 1002)
        self.assertEqual(service.data.n_users, 10405)
        self.assertTrue(all(handset_decision(d)[0] for d in service.documents.values()))
        self.assertEqual(service.data.directory.name, "handsets_only_v2")
        with self.assertRaisesRegex(ValueError, "No query-supervised"):
            runtime.recommender(joint="joint_retrieval")

    def test_http_uses_new_catalog_and_keeps_browser_history_separate(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as temporary:
            process = subprocess.Popen([sys.executable, "-B", str(ROOT / "src/run.py"), "chat_app", "--handsets",
                "--method", "legacy", "--recsys-method", "lightgcn", "--port", str(port),
                "--interaction-db", str(Path(temporary) / "profile.sqlite")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                url, client = f"http://127.0.0.1:{port}", requests.Session()
                for _ in range(150):
                    if process.poll() is not None:
                        self.fail(process.stdout.read().decode("utf-8", errors="replace"))
                    try:
                        if client.get(url, timeout=1).status_code == 200:
                            break
                    except requests.ConnectionError:
                        pass
                    time.sleep(.1)
                result = client.post(url + "/api/query", json={"query": "phone", "use_llm": False}, timeout=30)
                self.assertEqual(result.status_code, 200, result.text)
                body = result.json()
                self.assertTrue(body["results"])
                self.assertFalse(body["llm_enabled"])
                item = body["results"][0]["doc_id"]
                event = client.post(url + "/api/events", json={"kind": "save", "item_id": item}, timeout=10)
                self.assertEqual(event.status_code, 200, event.text)
                self.assertTrue(event.json()["profile"]["active_actions"][item]["save"])
                rejected = client.post(url + "/api/events", json={"kind": "save", "item_id": "B00005LLYR"}, timeout=10)
                self.assertEqual(rejected.status_code, 400)
                mixed_identity = client.post(url + "/api/query", json={"query": "phone", "use_llm": False, "user_index": 0}, timeout=10)
                self.assertEqual(mixed_identity.status_code, 400)
            finally:
                process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
