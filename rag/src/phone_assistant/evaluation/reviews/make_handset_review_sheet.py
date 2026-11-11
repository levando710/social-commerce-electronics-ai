"""Create a blank human catalog-review worksheet, never automatic human labels."""
import csv
import json
from pathlib import Path

from phone_assistant.data.phone_scope import handset_decision
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, read_jsonl, sha256


def main():
    split = ROOT / 'data/external/benchmarks/handsets_only_v2'
    data = InteractionData(split)
    output = ROOT / 'artifacts/evaluation/manual_handset_catalog_labels.csv'
    if output.exists():
        raise FileExistsError("Keep human edits; use a new worksheet version")
    documents = {d["doc_id"]: d for d in read_jsonl(split / "documents.jsonl")}
    products = {p["parent_asin"]: p for p in read_jsonl(split / "products.jsonl")}
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "title", "categories", "brand", "model", "storage", "ram", "battery", "image",
            "automatic_scope_reason", "human_product_type", "metadata_issue", "reviewer", "notes"])
        for item in data.item_ids:
            p, d = products[item], documents[item]
            details = p.get("details") or {}
            writer.writerow([item, p["title"], " > ".join(p.get("categories") or []), details.get("Brand") or details.get("Manufacturer"),
                details.get("Model Name") or details.get("Item model number"), details.get("Memory Storage Capacity"),
                details.get("RAM") or details.get("RAM Memory Installed Size"), details.get("Battery Capacity"),
                next(iter(d.get("product_image_paths", [])), ""), handset_decision(p)[1], "", "", "", ""])
    manifest = {"split_manifest_sha256": sha256(split / "manifest.json"), "source_sha256": sha256(Path(__file__)),
        "rows": data.n_items, "human_labels_filled": 0,
        "instructions": ["human_product_type: handset / accessory / other / uncertain; leave blank until reviewed.",
            "metadata_issue: optional short description; do not infer specs or condition from a catalog photograph.",
            "Automatic screening is not a human label. Do not use blanks as correct/irrelevant judgments.",
            "Keep frozen v2 unchanged. Corrections require a new catalog/split/features experiment version.",
            "Review catalog type independently of model outcomes; do not remove only failed test cases."]}
    (output.parent / "manual_handset_catalog_instructions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"file": str(output), "rows": data.n_items, "human_labels_filled": 0}))


if __name__ == "__main__":
    main()
