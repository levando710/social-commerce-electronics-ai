from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.collection import fetch_amazon_category_2023 as fetcher

CONTENT = {"Toy.csv": b"user_id,parent_asin,rating,timestamp\n", "meta_Toy.jsonl": b'{"parent_asin":"A"}\n'}


def fake_head(url, **kwargs):
    response = MagicMock()
    response.headers = {"Content-Length": str(len(CONTENT[url.rsplit("/", 1)[1]]))}
    return response


def fake_get(url, headers=None, **kwargs):
    data = CONTENT[url.rsplit("/", 1)[1]]
    start = int((headers or {}).get("Range", "bytes=0-").split("=")[1].rstrip("-"))
    response = MagicMock(status_code=206 if start else 200)
    response.iter_content.return_value = [data[start:]]
    response.__enter__.return_value = response
    return response


class FetchTests(unittest.TestCase):
    def run_fetch(self, folder, **kwargs):
        with patch.object(fetcher.requests, "head", side_effect=fake_head), \
                patch.object(fetcher.requests, "get", side_effect=fake_get):
            return fetcher.fetch("Toy", Path(folder), **kwargs)

    def test_download_records_hashes_and_a_second_call_is_a_verified_noop(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = self.run_fetch(folder)
            self.assertEqual(set(manifest["files"]), set(CONTENT))
            self.assertTrue(all(len(v["sha256"]) == 64 for v in manifest["files"].values()))
            with patch.object(fetcher.requests, "get") as get:
                self.run_fetch(folder)
                get.assert_not_called()

    def test_existing_file_of_wrong_size_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "Toy.csv").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "different size"):
                self.run_fetch(folder)

    def test_partial_download_resumes_and_is_size_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "Toy.csv.part").write_bytes(CONTENT["Toy.csv"][:5])
            manifest = self.run_fetch(folder)
            self.assertEqual((Path(folder) / "Toy.csv").read_bytes(), CONTENT["Toy.csv"])
            self.assertEqual(manifest["files"]["Toy.csv"]["bytes"], len(CONTENT["Toy.csv"]))

    def test_size_cap_refuses_before_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(fetcher.requests, "head") as head, patch.object(fetcher.requests, "get") as get:
                head.return_value.headers = {"Content-Length": str(5 * 1024 ** 3)}
                with self.assertRaisesRegex(ValueError, "cap"):
                    fetcher.fetch("Toy", Path(folder), max_gib=1)
                get.assert_not_called()

    def test_changed_file_after_manifest_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_fetch(folder)
            (Path(folder) / "Toy.csv").write_bytes(b"X" * len(CONTENT["Toy.csv"]))
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.run_fetch(folder)


if __name__ == "__main__":
    unittest.main()
