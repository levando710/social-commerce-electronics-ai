"""Reuse existing product photos by verified bytes; mark unavailable photos explicitly."""
from phone_assistant.common.project_layout import asset_path
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/handsets_only_v2')
    parser.add_argument("--work", type=Path, default=ROOT / 'artifacts/cache/handset_feature_work_v1')
    args = parser.parse_args()
    data = InteractionData(args.split)
    directory = args.work / "images"
    if directory.exists():
        raise FileExistsError("Image cache exists; keep its provenance and use a new work path")
    directory.mkdir(parents=True)
    docs = {d["doc_id"]: d for d in read_jsonl(args.split / "documents.jsonl")}
    records = []
    for item in data.item_ids:
        record = {"item_id": item, "status": "failed", "errors": ["No usable existing local photo; downloading deferred"]}
        for value in docs[item].get("product_image_paths", []):
            source = asset_path(value)
            if not source.is_file():
                continue
            try:
                with Image.open(source) as image:
                    image.verify()
                with Image.open(source) as image:
                    width, height, format = image.width, image.height, image.format
            except (OSError, ValueError):
                continue
            target = directory / f"{item}.img"
            digest = sha256(source)
            shutil.copyfile(source, target)
            if sha256(target) != digest:
                raise ValueError("Copied image bytes differ")
            record = {"item_id": item, "status": "ok", "file": target.name, "sha256": digest, "url": None,
                "width": width, "height": height, "format": format, "bytes": target.stat().st_size,
                "origin": "existing_local_product_photo", "source_file": str(source.resolve()),
                "seed_source_sha256": sha256(Path(__file__))}
            break
        (directory / f"{item}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    manifest = {"split_manifest_sha256": sha256(args.split / "manifest.json"),
        "documents_sha256": sha256(args.split / "documents.jsonl"), "source_sha256": sha256(Path(__file__)),
        "items": len(records), "available": sum(r["status"] == "ok" for r in records),
        "policy": "Actual decoded local product-photo bytes, no image generation or missing-image inference; remote downloads deferred."}
    (args.work / "image_seed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
