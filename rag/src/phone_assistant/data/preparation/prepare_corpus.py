"""Prepare product-level documents for the RAG demo."""
from phone_assistant.common.project_layout import MODULE_ROOT
from phone_assistant.common.project_layout import rebind_paths
import argparse
from collections import defaultdict
import json
from pathlib import Path


ROOT = MODULE_ROOT


DETAIL_KEYS = (
    "Brand", "Manufacturer", "Item model number", "Model Name", "Wireless Carrier",
    "Cellular Technology", "Wireless network technology", "Memory Storage Capacity",
    "Color", "Other camera features", "Connectivity technologies",
)


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_list(values, limit):
    result = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def resolve_image_base(data_dir):
    summary_path = data_dir / "processed" / "summary.json"
    if not summary_path.exists():
        return data_dir
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return data_dir
    source = summary.get("source_data")
    if source:
        return Path(rebind_paths(source))
    return data_dir


def image_paths(record, image_base):
    paths = []
    for asset in record.get("image_assets") or []:
        path = asset.get("path")
        if not path or asset.get("error"):
            continue
        full = image_base / path
        if full.exists():
            paths.append(str(full))
    return paths


def load_image_analysis(path):
    analyses = defaultdict(list)
    if not path or not path.exists():
        return analyses
    for row in read_jsonl(path):
        doc_id = row.get("doc_id")
        analysis = row.get("analysis")
        if doc_id and isinstance(analysis, dict):
            analyses[doc_id].append({
                "image_path": row.get("image_path"),
                "caption_vi": analysis.get("caption_vi"),
                "caption_en": analysis.get("caption_en"),
                "visible_product_type": analysis.get("visible_product_type"),
                "visible_brand_or_model": analysis.get("visible_brand_or_model"),
                "visual_tags": analysis.get("visual_tags") or [],
                "condition_tags": analysis.get("condition_tags") or [],
                "risk_tags": analysis.get("risk_tags") or [],
                "evidence": analysis.get("evidence") or [],
            })
    return analyses


def visual_text_items(items):
    parts = []
    for item in items or []:
        for key in ("caption_vi", "caption_en", "visible_product_type", "visible_brand_or_model"):
            if item.get(key):
                parts.append(str(item[key]))
        for key in ("visual_tags", "condition_tags", "risk_tags", "evidence"):
            parts.extend(str(value) for value in item.get(key) or [] if value)
    return compact_list(parts, 30)


def build_document(product, reviews, image_base, image_analyses=None):
    details = product.get("details") or {}
    selected_details = {
        key: details[key] for key in DETAIL_KEYS
        if isinstance(details, dict) and details.get(key)
    }
    review_items = []
    for review in reviews[:10]:
        text = str(review.get("text") or "").strip()
        if not text:
            continue
        review_items.append({
            "rating": review.get("rating"),
            "title": review.get("title"),
            "text": text,
            "verified_purchase": review.get("verified_purchase"),
            "helpful_vote": review.get("helpful_vote", 0),
        })
    text_parts = [
        f"Title: {product.get('title') or ''}",
        f"Brand/store: {product.get('store') or ''}",
        f"Categories: {' > '.join(product.get('categories') or [])}",
        f"Average rating: {product.get('average_rating')} from {product.get('rating_number')} ratings",
    ]
    features = compact_list(product.get("features"), 8)
    descriptions = compact_list(product.get("description"), 4)
    if features:
        text_parts.append("Features: " + " | ".join(features))
    if descriptions:
        text_parts.append("Description: " + " ".join(descriptions))
    if selected_details:
        text_parts.append(
            "Details: " + " | ".join(f"{key}: {value}" for key, value in selected_details.items())
        )
    if review_items:
        text_parts.append("Reviews:")
        for item in review_items:
            text_parts.append(
                f"- rating {item['rating']}; {item.get('title') or ''}; {item['text']}"
            )
    product_images = image_paths(product, image_base)
    review_images = []
    for review in reviews:
        review_images.extend(image_paths(review, image_base))
    doc_id = product.get("parent_asin") or product.get("record_id")
    visual_items = image_analyses.get(doc_id, []) if image_analyses else []
    visual_text = visual_text_items(visual_items)
    if visual_text:
        text_parts.append("Visual evidence: " + " | ".join(visual_text))
    return {
        "doc_id": doc_id,
        "parent_asin": product.get("parent_asin"),
        "title": product.get("title"),
        "brand": product.get("store"),
        "categories": product.get("categories") or [],
        "average_rating": product.get("average_rating"),
        "rating_number": product.get("rating_number"),
        "review_count": len(reviews),
        "product_image_paths": product_images[:5],
        "review_image_paths": review_images[:5],
        "details": selected_details,
        "reviews": review_items,
        "visual_evidence": visual_items,
        "text": "\n".join(text_parts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / 'data/collections/phones_filtered')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/runtime/catalog')
    parser.add_argument("--image-analysis", type=Path,
                        default=ROOT / 'artifacts/runtime/multimodal' / "image_analysis.jsonl")
    parser.add_argument("--require-review", action="store_true",
                        help="Keep only products with at least one review")
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    image_base = resolve_image_base(data)
    products = read_jsonl(data / "processed" / "products.jsonl")
    reviews = read_jsonl(data / "processed" / "reviews.jsonl")
    image_analyses = load_image_analysis(args.image_analysis.resolve())
    reviews_by_parent = defaultdict(list)
    for review in reviews:
        reviews_by_parent[review.get("parent_asin")].append(review)

    documents = []
    for product in products:
        parent = product.get("parent_asin")
        product_reviews = reviews_by_parent.get(parent, [])
        if args.require_review and not product_reviews:
            continue
        documents.append(build_document(product, product_reviews, image_base, image_analyses))

    output.mkdir(parents=True, exist_ok=True)
    target = output / "documents.jsonl"
    with target.with_suffix(".part").open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    target.with_suffix(".part").replace(target)

    summary = {
        "source_data": str(data),
        "image_base": str(image_base),
        "documents": len(documents),
        "documents_with_reviews": sum(doc["review_count"] > 0 for doc in documents),
        "documents_with_product_images": sum(bool(doc["product_image_paths"]) for doc in documents),
        "documents_with_review_images": sum(bool(doc["review_image_paths"]) for doc in documents),
        "documents_with_visual_evidence": sum(bool(doc["visual_evidence"]) for doc in documents),
        "image_analysis_source": str(args.image_analysis.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
