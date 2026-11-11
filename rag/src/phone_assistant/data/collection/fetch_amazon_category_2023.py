"""Download one Amazon Reviews 2023 category (5-core ratings + item metadata) with provenance.

Files come from the public ``McAuley-Lab/Amazon-Reviews-2023`` dataset on Hugging Face.  Each file is
resumed if a partial copy exists, checked against the size the server reports, and recorded with its
SHA-256 in ``manifest.json``.  A byte-cap refuses huge categories (Clothing is ~18 GB) unless raised
on purpose.  Files that already exist are verified and adopted rather than downloaded again.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from phone_assistant.recommendation.recsys_data import ROOT, sha256

BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
SOURCE_PAGE = "https://amazon-reviews-2023.github.io/"
DEFAULT_CATEGORY = "Baby_Products"
CHUNK_BYTES = 1024 ** 2
DEFAULT_MAX_GIB = 2
HTTP_TIMEOUT_SECONDS = (15, 60)


def file_plan(category):
    """name -> remote path inside the dataset repository."""
    return {f"{category}.csv": f"benchmark/5core/rating_only/{category}.csv",
            f"meta_{category}.jsonl": f"raw/meta_categories/meta_{category}.jsonl"}


def remote_size(url, headers=None):
    response = requests.head(url, headers=headers, allow_redirects=True, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def download(url, path, expected_bytes, extra_headers=None):
    """Resume-capable streaming download; the result must match the server-reported size."""
    part = path.with_name(path.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = dict(extra_headers or {})
    if start:
        headers["Range"] = f"bytes={start}-"
    with requests.get(url, headers=headers, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        if start and response.status_code != requests.codes.partial_content:
            raise ValueError("Server ignored the resume range; partial file kept, not appended")
        with part.open("ab" if start else "wb") as handle:
            for chunk in response.iter_content(CHUNK_BYTES):
                handle.write(chunk)
    if part.stat().st_size != expected_bytes:
        raise ValueError(f"{path.name}: downloaded {part.stat().st_size} bytes, server reports {expected_bytes}")
    part.replace(path)


def fetch(category, output, max_gib=DEFAULT_MAX_GIB):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    plan = file_plan(category)
    sizes = {name: remote_size(f"{BASE_URL}/{remote}") for name, remote in plan.items()}
    if sum(sizes.values()) > max_gib * 1024 ** 3:
        raise ValueError(f"{category} needs {sum(sizes.values()) / 1024 ** 3:.1f} GiB, above the {max_gib} GiB cap; "
                         "raise --max-gib only if that is intended")
    started = time.perf_counter()
    files = {}
    for name, remote in plan.items():
        path = output / name
        if path.exists():
            if path.stat().st_size != sizes[name]:
                raise ValueError(f"{name} exists with a different size than the server reports; remove it or investigate")
        else:
            download(f"{BASE_URL}/{remote}", path, sizes[name])
        files[name] = {"url": f"{BASE_URL}/{remote}", "bytes": sizes[name], "sha256": sha256(path)}
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        if recorded != files:
            raise ValueError("Files no longer match the recorded manifest")
    else:
        manifest = {"dataset": "McAuley-Lab/Amazon-Reviews-2023", "category": category, "source_page": SOURCE_PAGE,
                    "files": files, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seconds": round(time.perf_counter() - started, 1), "source_sha256": sha256(Path(__file__)),
                    "limitations": ["Public research dataset; check its license before redistribution.",
                                    "Size is verified against the server; no content audit is performed here."]}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-gib", type=float, default=DEFAULT_MAX_GIB)
    args = parser.parse_args()
    output = args.output or ROOT / f"data/external/amazon_2023_{args.category.lower()}"
    print(json.dumps(fetch(args.category, output, args.max_gib), indent=2))


if __name__ == "__main__":
    main()
