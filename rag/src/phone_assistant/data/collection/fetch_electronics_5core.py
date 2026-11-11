"""Download the official Electronics rating archive into its own versioned source folder."""
import argparse
import json
from pathlib import Path
import time

import requests

from phone_assistant.recommendation.recsys_data import ROOT, sha256

SOURCE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Electronics.csv.gz"
SOURCE_PAGE = "https://amazon-reviews-2023.github.io/data_processing/5core.html"
ARCHIVE_NAME = "Electronics.csv.gz"
CHUNK_BYTES = 1024 ** 2
MAXIMUM_ARCHIVE_BYTES = 512 * CHUNK_BYTES
PROGRESS_INTERVAL_BYTES = 16 * CHUNK_BYTES
HTTP_TIMEOUT_SECONDS = (15, 30)
GZIP_MAGIC = b"\x1f\x8b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/amazon_5core_electronics')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path, manifest_path = args.output / ARCHIVE_NAME, args.output / "manifest.json"
    if path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["url"] != SOURCE_URL or sha256(path) != manifest["sha256"]:
            raise ValueError("Frozen source mismatch")
        print(json.dumps(manifest, indent=2)); return
    temporary = path.with_suffix(".gz.part")
    start = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    started = time.perf_counter()
    with requests.get(SOURCE_URL, headers=headers, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        if start and (response.status_code != requests.codes.partial_content or
                      not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
            raise ValueError("Server ignored resume range; partial file retained, do not append")
        size, last_progress = start, start
        with temporary.open("ab" if start else "wb") as handle:
            for chunk in response.iter_content(CHUNK_BYTES):
                size += len(chunk)
                if size > MAXIMUM_ARCHIVE_BYTES:
                    raise ValueError("Compressed archive exceeds configured 512 MiB budget")
                handle.write(chunk)
                if size - last_progress >= PROGRESS_INTERVAL_BYTES:
                    print(f"Electronics archive: {size / CHUNK_BYTES:.1f} MiB", flush=True)
                    last_progress = size
    with temporary.open("rb") as handle:
        if handle.read(len(GZIP_MAGIC)) != GZIP_MAGIC:
            raise ValueError("Source is not gzip data")
    digest = sha256(temporary)
    manifest = {"url": SOURCE_URL, "source_page": SOURCE_PAGE, "category": "Electronics",
        "sha256": digest, "bytes": size, "download_seconds": time.perf_counter() - started,
        "source_sha256": sha256(Path(__file__)), "status": "downloaded_archive_not_full_schema_audited",
        "limitations": ["IDs/ratings/timestamps only; metadata/images must be joined separately.",
            "Archive-wide row/user/item counts are not computed here; split preparation records scanned rows."]}
    # A downloaded archive is finalized only with its provenance manifest.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
