import argparse
from contextlib import redirect_stdout
import io
import importlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
collector = importlib.import_module("phone_assistant.data.collection.collect_amazon")


class CollectorTests(unittest.TestCase):
    def test_http_resume_and_rejected_ignored_range(self):
        class Response:
            status_code = 206
            headers = {"Content-Range": "bytes 3-5/6", "ETag": '"version1"'}

            def __enter__(self):
                self.raw = io.BytesIO(b"{}\n")
                return self

            def __exit__(self, *args):
                self.raw.close()

            def raise_for_status(self):
                pass

        response = Response()
        def get(url, **kwargs):
            self.assertEqual(kwargs["headers"]["Range"], "bytes=3-")
            return response
        with patch.dict("sys.modules", {"requests": types.SimpleNamespace(get=get)}):
            with collector.source_stream("https://example.org/data", 3, '"version1"') as (stream, _):
                self.assertEqual(stream.readline(), b"{}\n")
            response.status_code = 200
            with self.assertRaises(ValueError):
                with collector.source_stream("https://example.org/data", 3, '"version1"'):
                    pass
            response.status_code = 206
            with self.assertRaises(ValueError):
                with collector.source_stream("https://example.org/data", 3, '"changed"'):
                    pass

    def test_phone_and_accessory(self):
        self.assertTrue(collector.phone_decision({"title": "Samsung Galaxy S22 128GB Unlocked"})[0])
        self.assertFalse(collector.phone_decision({"title": "Case for Samsung Galaxy S22"})[0])
        self.assertFalse(collector.phone_decision({"title": "iPhone 13 replacement screen"})[0])
        self.assertFalse(collector.phone_decision({"title": "Google Pixel Watch 2"})[0])
        self.assertTrue(collector.phone_decision({
            "title": "Apple iPhone 13", "description": ["Includes charging cable and case"]})[0])
        self.assertTrue(collector.phone_decision({
            "title": "Apple iPhone 13 with case",
            "categories": ["Cell Phones & Accessories", "Cell Phones"],
        })[0])

    def test_real_audit_noise_is_rejected(self):
        noisy_rows = [
            {
                "title": "Google Cardboard Virtual Reality Headset for Samsung Galaxy Note 3 iPhone 6 Plus",
                "categories": ["Cell Phones & Accessories", "Accessories",
                               "Virtual Reality (VR) Headsets"],
            },
            {
                "title": "Sierra Wireless AC595U EVDO USB Modem (Sprint)",
                "categories": ["Cell Phones & Accessories", "Cell Phones"],
            },
            {
                "title": "Aspercreme Fast Acting Heat Arthritis Pain Relieving Gel",
                "categories": ["Cell Phones & Accessories", "Cell Phones"],
            },
            {
                "title": "Tempered Glass Screen Protector Compatible with iPhone 13",
                "categories": ["Cell Phones & Accessories", "Accessories", "Screen Protectors"],
            },
            {
                "title": "Silicon Skin Blue - Motorola Razr V3, V3c, V3i",
                "categories": ["Cell Phones & Accessories"],
            },
            {
                "title": "BlackBerry OEM 8520 & 9300 Black Skin",
                "categories": ["Cell Phones & Accessories"],
            },
            {
                "title": "Mini All-in-1 Memory Card Reader Compatible with Sony",
                "categories": ["Cell Phones & Accessories"],
            },
            {
                "title": "CoverON Slim Hard Case for HTC One Mini with Cover Removal Tool",
                "categories": ["Cell Phones & Accessories"],
            },
            {
                "title": "LG Ultimate 2 Case, Hard Cover with Kickstand | Includes Screen Protector",
                "categories": ["Cell Phones & Accessories"],
            },
            {
                "title": "ZTE Grand X Max Case with Kickstand and Belt Clip | Includes Screen Protector",
                "categories": ["Cell Phones & Accessories"],
            },
        ]
        for row in noisy_rows:
            self.assertFalse(collector.phone_decision(row)[0], row["title"])

    def test_older_phone_titles_are_accepted(self):
        rows = [
            {"title": "Palm Treo 600 GSM Cell Phone", "categories": ["Cell Phones & Accessories", "Cell Phones"]},
            {"title": "Motorola RAZR V3 Unlocked GSM Phone", "categories": ["Cell Phones & Accessories", "Cell Phones"]},
            {"title": "Sony Ericsson W850i Mobile Phone", "categories": ["Cell Phones & Accessories", "Cell Phones"]},
            {"title": "Nokia N79 Unlocked Phone", "categories": ["Cell Phones & Accessories", "Cell Phones"]},
            {"title": "Pantech Breeze 3 Basic Flip Phone GSM Unlocked", "categories": ["Cell Phones & Accessories", "Cell Phones"]},
            {"title": "BLU Q53i Samba Phone with QWERTY Keypad", "categories": ["Cell Phones & Accessories", "Cell Phones"]},
        ]
        for row in rows:
            self.assertTrue(collector.phone_decision(row)[0], row["title"])

    def test_unknown_is_not_assumed_phone(self):
        self.assertEqual(collector.phone_decision({"title": "Mystery device"}),
                         (False, "needs_manual_review"))

    def test_image_schemas(self):
        url = "https://m.media-amazon.com/images/a.jpg"
        for images in ([{"hi_res": None, "large": url}],
                       {"hi_res": [None], "large": [url]},
                       [{"large_image_url": url}],
                       [{"large": url}, {"large": url}]):
            self.assertEqual(collector.image_urls(images), [url])

    def test_resume_join_dedup_and_export(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            meta = folder / "meta.jsonl"
            reviews = folder / "review.jsonl"
            meta.write_text("\n".join(json.dumps(row) for row in [
                {"parent_asin": "A", "title": "iPhone 13"},
                {"parent_asin": "X", "title": "iPhone 13 case"},
                {"parent_asin": "B", "title": "Google Pixel 7"}]) + "\n", encoding="utf-8")
            review = {"parent_asin": "A", "asin": "VARIANT", "text": "Battery lasts all day", "rating": 5}
            reviews.write_text("\n".join(json.dumps(row) for row in [review, review,
                {"parent_asin": "X", "text": "Accessory review"},
                {"parent_asin": "B", "text": "Screen is good"}]) + "\n", encoding="utf-8")
            args = argparse.Namespace(metadata_source=str(meta), reviews_source=str(reviews),
                                      output=folder / "out", max_products=10,
                                      scan_limit=2, reviews_per_product=20)
            db = collector.connect(args.output / "raw" / "amazon.sqlite3")
            try:
                collector.collect(db, "metadata", args)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM products").fetchone()[0], 1)
                collector.collect(db, "metadata", args)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM products").fetchone()[0], 2)
                collector.collect(db, "reviews", args)
                collector.collect(db, "reviews", args)
                collector.collect(db, "reviews", args)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 2)
                collector.export(db, args)
                report = json.loads((args.output / "processed" / "summary.json").read_text())
                self.assertEqual(report["products_with_reviews"], 2)
                exported = [json.loads(s) for s in
                            (args.output / "processed" / "reviews.jsonl").read_text().splitlines()]
                self.assertEqual({r["parent_asin"] for r in exported}, {"A", "B"})
            finally:
                db.close()

    def test_changed_local_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_bytes(b'{}\n')
            with collector.source_stream(str(source), 0, None) as (_, identity):
                pass
            source.write_bytes(b'{}\n{}\n')
            with self.assertRaises(ValueError):
                with collector.source_stream(str(source), 3, identity):
                    pass


if __name__ == "__main__":
    unittest.main()
