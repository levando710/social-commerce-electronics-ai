"""Download and unpack the MIND-small news-recommendation dataset from its gated Hugging Face mirror.

The original Azure host no longer allows public access.  The Hugging Face copy (``yjw1029/MIND``) needs
an account that accepted the dataset terms (Microsoft Research data license, see
https://github.com/msnews/MIND).  The read token comes from ``HF_TOKEN`` in the environment or ``rag/.env``
and is only sent to huggingface.co; ``requests`` drops it when redirected to another host.  Archives are
size-checked against the server, hashed into ``manifest.json`` and extracted without path traversal.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from phone_assistant.data.collection.fetch_amazon_category_2023 import download, remote_size
from phone_assistant.llm.gemini_client import get_secret
from phone_assistant.recommendation.recsys_data import ROOT, sha256

BASE_URL = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main"
LICENSE_PAGE = "https://github.com/msnews/MIND"
ARCHIVES = ("MINDsmall_train.zip", "MINDsmall_dev.zip")
EXPECTED_MEMBERS = {"news.tsv", "behaviors.tsv"}


def safe_extract(archive, target):
    """Extract only plain file members that stay inside ``target``."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if target != destination and target not in destination.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        bundle.extractall(target)
        return sorted(member.filename for member in bundle.infolist() if not member.is_dir())


def fetch(output, token):
    if not token:
        raise ValueError("HF_TOKEN not found; create a read token after accepting the MIND terms on Hugging Face")
    auth = {"Authorization": f"Bearer {token}"}
    output.mkdir(parents=True, exist_ok=True)
    started, files = time.perf_counter(), {}
    for name in ARCHIVES:
        url, path = f"{BASE_URL}/{name}", output / name
        size = remote_size(url, auth)
        if path.exists():
            if path.stat().st_size != size:
                raise ValueError(f"{name} exists with a different size than the server reports")
        else:
            download(url, path, size, auth)
        members = safe_extract(path, output)  # members already carry their own top folder
        if not EXPECTED_MEMBERS <= {Path(m).name for m in members}:
            raise ValueError(f"{name} lacks the expected news/behaviors tables")
        files[name] = {"url": url, "bytes": size, "sha256": sha256(path), "members": members}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["files"] != files:
            raise ValueError("Files no longer match the recorded manifest")
    else:
        manifest_path.write_text(json.dumps({
            "dataset": "MIND-small (Microsoft News Dataset), Hugging Face mirror yjw1029/MIND", "license_page": LICENSE_PAGE,
            "files": files, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.perf_counter() - started, 1), "source_sha256": sha256(Path(__file__)),
            "limitations": ["Gated mirror; terms are those of the Microsoft Research data license, which was not read here.",
                            "English news text only: no product images or specifications."]}, indent=2), encoding="utf-8")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/mind_small")
    args = parser.parse_args()
    print(json.dumps(fetch(args.output, get_secret("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")), indent=2))


if __name__ == "__main__":
    main()
