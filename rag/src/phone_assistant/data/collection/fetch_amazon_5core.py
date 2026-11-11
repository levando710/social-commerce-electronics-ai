"""Download and stream-audit official phone 5-core IDs without loading the CSV into RAM."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import requests

from phone_assistant.recommendation.recsys_data import ROOT, sha256

URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Cell_Phones_and_Accessories.csv.gz"
SOURCE = "https://amazon-reviews-2023.github.io/data_processing/5core.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/amazon_5core')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "Cell_Phones_and_Accessories.csv.gz"
    manifest_path = args.output / "manifest.json"
    if path.exists():
        if not manifest_path.exists() or sha256(path) != json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError("Existing file has no matching manifest; use another output path")
    else:
        temporary = path.with_suffix(".gz.part")
        size = 0
        digest = hashlib.sha256()
        with requests.get(URL, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 512 * 1024 * 1024:
                        raise ValueError("Compressed download exceeds 512 MiB")
                    output.write(chunk)
                    digest.update(chunk)
        temporary.replace(path)
        manifest_path.write_text(json.dumps({"url": URL, "source_page": SOURCE, "sha256": digest.hexdigest(),
                                             "bytes": size}, indent=2), encoding="utf-8")
    users, items, ratings = set(), set(), {}
    rows, min_ts, max_ts, invalid = 0, None, None, 0
    # Read bytes with an explicit decompression limit, then decode one line at a time.
    with gzip.open(path, "rb") as compressed:
        total_bytes = 0
        def lines():
            nonlocal total_bytes
            for line in compressed:
                total_bytes += len(line)
                if total_bytes > 1024 * 1024 * 1024:
                    raise ValueError("Uncompressed CSV exceeds 1 GiB")
                yield line.decode("utf-8-sig")
        reader = csv.DictReader(lines())
        if reader.fieldnames != ["user_id", "parent_asin", "rating", "timestamp"]:
            raise ValueError(f"Unexpected source schema: {reader.fieldnames}")
        for row in reader:
            rows += 1
            timestamp, rating = int(row["timestamp"]), float(row["rating"])
            if not row["user_id"] or not row["parent_asin"] or not 1 <= rating <= 5 or timestamp <= 0:
                invalid += 1
                continue
            users.add(row["user_id"])
            items.add(row["parent_asin"])
            ratings[str(rating)] = ratings.get(str(rating), 0) + 1
            min_ts = timestamp if min_ts is None else min(min_ts, timestamp)
            max_ts = timestamp if max_ts is None else max(max_ts, timestamp)
    result = {**json.loads(manifest_path.read_text(encoding="utf-8")), "rows": rows, "users": len(users),
              "items": len(items), "ratings": ratings, "timestamp_range": [min_ts, max_ts], "invalid_rows": invalid,
              "status": "downloaded_and_schema_audited_not_yet_split_or_trained",
              "limitations": ["Pure IDs/ratings/timestamps: no product text, images or dialogues in this file.",
                  "5-core filtering applies before positive-rating filtering and subsequent data splits.",
                  "Must exclude held-out user-target pairs and nonpast benchmark histories before joint use.",
                  "User/item overlap and duplicate pairs not audited in this streaming pass."]}
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
