import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.preparation import prepare_amazon_category_recsys as prep


def image_product(asin):
    return {"parent_asin": asin, "title": f"Product {asin}", "store": "S", "features": [], "description": [],
            "categories": ["Baby"], "details": {}, "images": [{"variant": "MAIN", "large": f"https://m.media-amazon.com/{asin}.jpg"}]}


class AmazonCategoryRecsysTests(unittest.TestCase):
    def test_duplicate_pair_keeps_newer_event_and_negative_ratings_are_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "r.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
                writer.writerows([["u", "a", 5.0, 10], ["u", "a", 4.0, 20], ["u", "b", 2.0, 5]])
            result = prep.read_positives(path)
        self.assertEqual(result["u"], {"a": (20, 4.0)})

    def test_k_core_is_stable_and_respects_both_thresholds(self):
        data = {f"u{n}": {f"i{k}": (k, 5.0) for k in range(5)} for n in range(5)}
        data["lonely"] = {"i0": (1, 5.0), "rare": (2, 5.0)}
        core = prep.k_core(data)
        self.assertNotIn("lonely", core)
        self.assertTrue(all("rare" not in items for items in core.values()))
        self.assertEqual(prep.k_core(core), core)

    def test_leave_last_out_orders_by_time_and_never_overlaps(self):
        data = {"u": {f"i{k}": (100 - k, 5.0) for k in range(5)}}
        split = prep.leave_last_out(data)
        self.assertEqual([r["item_id"] for r in split["test"]], ["i0"])
        self.assertEqual([r["item_id"] for r in split["validation"]], ["i1"])
        self.assertEqual(sorted(r["item_id"] for r in split["train"]), ["i2", "i3", "i4"])

    def test_line_separator_characters_in_text_survive_the_shared_reader(self):
        from phone_assistant.recommendation.recsys_data import read_jsonl
        text = "a b c\x0bd\x85e"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "p.jsonl"
            prep.write_jsonl(path, [{"title": text}, {"title": "next"}])
            rows = read_jsonl(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], text)

    def test_sampling_is_deterministic(self):
        items = {f"i{n}" for n in range(100)}
        self.assertEqual(prep.sample_items(items, 10, 1), prep.sample_items(items, 10, 1))
        self.assertNotEqual(prep.sample_items(items, 10, 1), prep.sample_items(items, 10, 2))

    def test_end_to_end_split_loads_with_the_frozen_loader(self):
        from phone_assistant.recommendation.recsys_data import InteractionData
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asins = [f"A{n}" for n in range(8)]
            with (root / "r.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
                for user in range(8):
                    for step, asin in enumerate(asins[:7] if user % 2 else asins[1:]):
                        writer.writerow([f"u{user}", asin, 5.0, step + 1])
            (root / "m.jsonl").write_text("".join(json.dumps(image_product(a)) + "\n" for a in asins), encoding="utf-8")
            summary = prep.prepare(root / "r.csv", root / "m.jsonl", root / "out", sampled_items=100, seed=1)
            data = InteractionData(root / "out")
            self.assertEqual(data.n_users, summary["users"])
            self.assertEqual(len(data.pairs["test"]), summary["users"])
            self.assertIn("popularity_metrics.json", data.manifest["artifacts_sha256"])
            with self.assertRaises(FileExistsError):
                prep.prepare(root / "r.csv", root / "m.jsonl", root / "out")
            # Retrofit path: regenerating the file for an existing split reproduces the same content.
            original = (root / "out" / "popularity_metrics.json").read_text(encoding="utf-8")
            (root / "out" / "popularity_metrics.json").unlink()
            prep.write_popularity_sidecar(root / "out")
            self.assertEqual((root / "out" / "popularity_metrics.json").read_text(encoding="utf-8"), original)
            with self.assertRaises(FileExistsError):
                prep.write_popularity_sidecar(root / "out")


if __name__ == "__main__":
    unittest.main()
