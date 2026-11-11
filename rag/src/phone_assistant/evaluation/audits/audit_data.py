"""Offline audit of the exported RAG corpus. Does not modify source data."""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image

ROOT = MODULE_ROOT


def read_records(data, name):
    with (data / "processed" / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def category_flag(product):
    categories = product.get("categories") or []
    accessory_nodes = {"Accessories", "Cases, Holsters & Sleeves", "iPhone Accessories",
                       "SIM Cards & Prepaid Minutes"}
    # The root includes Accessories for every product; inspect descendants only.
    return bool(set(categories[1:]) & accessory_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data",
                        help="Corpus directory containing processed/, raw/, images/")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports",
                        help="Directory for data_audit.json and product_audit.csv")
    args = parser.parse_args()
    data = args.data.resolve()
    products = read_records(data, "products.jsonl")
    reviews = read_records(data, "reviews.jsonl")
    ids = {p["parent_asin"] for p in products}
    by_parent = Counter(r.get("parent_asin") for r in reviews)
    image_results = {}
    hashes = defaultdict(list)
    for path in (data / "images").glob("*.jpg"):
        try:
            with Image.open(path) as image:
                size = image.size
                image.verify()
            image_results[path.relative_to(data).as_posix()] = {"size": size, "valid": True}
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
        except (OSError, ValueError) as exc:
            image_results[path.relative_to(data).as_posix()] = {"valid": False, "error": str(exc)}
    def available(record):
        return [a for a in record.get("image_assets", [])
                if image_results.get(a.get("path"), {}).get("valid")]
    flagged = [p for p in products if category_flag(p)]
    unflagged = [p for p in products if not category_flag(p)]
    missing_fields = {field: sum(not p.get(field) for p in products)
                      for field in ("title", "description", "features", "details", "categories", "images", "videos")}
    title_counts = Counter(p.get("title", "").strip().casefold() for p in products)
    stats = {
        "products": len(products), "reviews": len(reviews),
        "duplicate_parent_ids": len(products) - len(ids),
        "duplicate_review_record_ids": len(reviews) - len({r["record_id"] for r in reviews}),
        "orphan_reviews": sum(r.get("parent_asin") not in ids for r in reviews),
        "products_with_reviews": sum(by_parent[p] > 0 for p in ids),
        "products_without_reviews": sum(by_parent[p] == 0 for p in ids),
        "review_count_distribution": dict(sorted(Counter(by_parent[p] for p in ids).items())),
        "rating_distribution": dict(Counter(r.get("rating") for r in reviews)),
        "products_flagged_accessory_category": len(flagged),
        "reviews_on_flagged_products": sum(by_parent[p["parent_asin"]] for p in flagged),
        "unflagged_products": len(unflagged),
        "unflagged_with_reviews": sum(by_parent[p["parent_asin"]] > 0 for p in unflagged),
        "unflagged_with_reviews_and_local_images": sum(
            by_parent[p["parent_asin"]] > 0 and bool(available(p)) for p in unflagged),
        "missing_product_fields": missing_fields,
        "product_leaf_categories": dict(Counter((p.get("categories") or ["UNKNOWN"])[-1] for p in products)),
        "products_with_local_images": sum(bool(available(p)) for p in products),
        "reviews_with_local_images": sum(bool(available(r)) for r in reviews),
        "reviews_with_image_urls": sum(bool(r.get("image_assets")) for r in reviews),
        "local_image_files": len(image_results),
        "invalid_local_images": sum(not r["valid"] for r in image_results.values()),
        "missing_referenced_image_files": sum(
            bool(a.get("path")) and a["path"] not in image_results
            for r in products + reviews for a in r.get("image_assets", [])),
        "small_images_under_224_on_short_side": sum(
            r["valid"] and min(r["size"]) < 224 for r in image_results.values()),
        "exact_duplicate_image_groups": sum(len(paths) > 1 for paths in hashes.values()),
        "exact_duplicate_image_excess_files": sum(len(paths) - 1 for paths in hashes.values()),
        "duplicate_title_groups": sum(n > 1 for n in title_counts.values()),
        "review_text_under_20_characters": sum(len(str(r.get("text") or "").strip()) < 20 for r in reviews),
        "duplicate_review_text_excess": sum(n - 1 for n in Counter(
            str(r.get("text") or "").strip().casefold() for r in reviews).values()),
    }
    with sqlite3.connect((data / "raw" / "amazon.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        stats["database_integrity"] = db.execute("PRAGMA quick_check").fetchone()[0]
        stats["failed_downloads"] = [dict(zip(("url", "error"), row)) for row in
                                      db.execute("SELECT url,error FROM images WHERE error IS NOT NULL")]
        stats["db_counts"] = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                              for t in ("products", "reviews", "images")}
    report = args.report_dir.resolve()
    report.mkdir(exist_ok=True)
    (report / "data_audit.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with (report / "product_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parent_asin", "title", "categories", "accessory_category_flag",
                         "review_count", "local_image_count", "store"])
        for p in products:
            writer.writerow([p["parent_asin"], p["title"], " > ".join(p.get("categories") or []),
                             category_flag(p), by_parent[p["parent_asin"]], len(available(p)), p.get("store")])
    print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
