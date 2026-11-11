"""Summarize multimodal coverage for the RAG dataset."""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = MODULE_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(count, total):
    return round(count / total * 100, 2) if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--image-analysis", type=Path,
                        default=ROOT / 'artifacts/runtime/multimodal' / "image_analysis.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / 'artifacts/evaluation' / "results" / "multimodal_coverage.json")
    args = parser.parse_args()

    docs = read_jsonl(args.corpus.resolve())
    raw_image_rows = read_jsonl(args.image_analysis.resolve())
    latest_by_image = {}
    for row in raw_image_rows:
        key = (row.get("doc_id"), row.get("image_path"))
        if key[0] and key[1]:
            latest_by_image[key] = row
    image_rows = list(latest_by_image.values())
    total = len(docs)
    image_success = [row for row in image_rows if row.get("analysis")]
    image_failed = [row for row in image_rows if not row.get("analysis")]

    visual_types = Counter()
    visual_tags = Counter()
    condition_tags = Counter()
    risk_tags = Counter()
    for row in image_success:
        analysis = row.get("analysis") or {}
        visual_types.update([analysis.get("visible_product_type") or "unknown"])
        visual_tags.update(analysis.get("visual_tags") or [])
        condition_tags.update(analysis.get("condition_tags") or [])
        risk_tags.update(analysis.get("risk_tags") or [])

    report = {
        "documents": total,
        "text_documents": total,
        "documents_with_reviews": sum(int(doc.get("review_count") or 0) > 0 for doc in docs),
        "documents_with_product_images": sum(bool(doc.get("product_image_paths")) for doc in docs),
        "documents_with_review_images": sum(bool(doc.get("review_image_paths")) for doc in docs),
        "documents_with_visual_evidence": sum(bool(doc.get("visual_evidence")) for doc in docs),
        "image_analysis_raw_rows": len(raw_image_rows),
        "image_analysis_unique_images": len(image_rows),
        "image_analysis_rows": len(image_rows),
        "image_analysis_success": len(image_success),
        "image_analysis_failed": len(image_failed),
        "coverage_percent": {
            "reviews": pct(sum(int(doc.get("review_count") or 0) > 0 for doc in docs), total),
            "product_images": pct(sum(bool(doc.get("product_image_paths")) for doc in docs), total),
            "review_images": pct(sum(bool(doc.get("review_image_paths")) for doc in docs), total),
            "visual_evidence": pct(sum(bool(doc.get("visual_evidence")) for doc in docs), total),
        },
        "visual_product_types": dict(visual_types.most_common()),
        "top_visual_tags": dict(visual_tags.most_common(20)),
        "top_condition_tags": dict(condition_tags.most_common(20)),
        "top_risk_tags": dict(risk_tags.most_common(20)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
