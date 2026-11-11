from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.collection import fetch_mind_small as mind


class MindFetchTests(unittest.TestCase):
    def test_extraction_keeps_files_inside_the_target(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "a.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("Part/news.tsv", "n\n")
                bundle.writestr("Part/behaviors.tsv", "b\n")
            members = mind.safe_extract(archive, Path(folder) / "out")
            self.assertEqual(members, ["Part/behaviors.tsv", "Part/news.tsv"])
            self.assertTrue((Path(folder) / "out/Part/news.tsv").is_file())

    def test_path_traversal_member_is_rejected_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "evil.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "x")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                mind.safe_extract(archive, Path(folder) / "out")
            self.assertFalse((Path(folder) / "escape.txt").exists())

    def test_missing_token_stops_before_any_request(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            mind.fetch(Path(tempfile.gettempdir()) / "mind_never_created", None)


if __name__ == "__main__":
    unittest.main()
