"""Export a stricter phone-only view from an already collected corpus."""
from phone_assistant.common.project_layout import MODULE_ROOT, source_path
import argparse
import csv
import importlib.util
import json
from pathlib import Path


ROOT = MODULE_ROOT


def load_collector():
    spec = importlib.util.spec_from_file_location("collector", source_path('collect_amazon.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".part").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.with_suffix(".part").replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Collected corpus directory, for example data_phone_clean_v2")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for filtered processed files")
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    collector = load_collector()

    products = read_jsonl(data / "processed" / "products.jsonl")
    reviews = read_jsonl(data / "processed" / "reviews.jsonl")

    kept_products = []
    rejected_rows = []
    keep_ids = set()
    reason_counts = {}
    for row in products:
        keep, reason = collector.phone_decision(row)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        parent = row.get("parent_asin")
        if keep:
            kept_products.append(row)
            keep_ids.add(parent)
        else:
            rejected_rows.append({
                "parent_asin": parent,
                "title": row.get("title"),
                "categories": " > ".join(row.get("categories") or []),
                "reason": reason,
            })

    kept_reviews = [row for row in reviews if row.get("parent_asin") in keep_ids]
    products_with_reviews = {row.get("parent_asin") for row in kept_reviews}

    processed = output / "processed"
    write_jsonl(processed / "products.jsonl", kept_products)
    write_jsonl(processed / "reviews.jsonl", kept_reviews)
    with (processed / "rejected_products.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parent_asin", "title", "categories", "reason"])
        writer.writeheader()
        writer.writerows(rejected_rows)

    summary = {
        "source_data": str(data),
        "products_before": len(products),
        "products_after": len(kept_products),
        "products_rejected": len(rejected_rows),
        "reviews_before": len(reviews),
        "reviews_after": len(kept_reviews),
        "products_with_reviews": len(products_with_reviews),
        "decision_reasons": reason_counts,
        "note": "Image paths in image_assets still point to paths relative to the source corpus directory.",
    }
    (processed / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
