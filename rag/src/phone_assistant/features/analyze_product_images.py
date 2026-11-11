"""Analyze product images with Gemini Vision and cache multimodal evidence.

This is an optional step for the Multimodal RAG pipeline. It does not scrape data
and does not train a model. Use --limit to control API usage.
"""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageStat

from phone_assistant.llm.gemini_client import generate_image_analysis


ROOT = MODULE_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_keys(path, retry_errors=False):
    done = set()
    for row in read_jsonl(path):
        if row.get("doc_id") and row.get("image_path"):
            if row.get("analysis") or not retry_errors:
                done.add((row["doc_id"], row["image_path"]))
    return done


def infer_product_type(title):
    text = str(title or "").lower()
    accessory_terms = (
        "case", "cover", "screen protector", "charger", "cable", "battery door",
        "holster", "belt clip", "skin", "protector", "mount", "adapter",
    )
    phone_terms = (
        "phone", "smartphone", "iphone", "galaxy", "pixel", "nokia", "motorola",
        "blackberry", "xperia", "oneplus", "redmi", "oppo", "vivo", "realme",
    )
    if any(term in text for term in accessory_terms):
        return "phone_accessory"
    if any(term in text for term in phone_terms):
        return "phone"
    return "unknown"


def bucket(value, low, high):
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "medium"


def dominant_color_name(rgb):
    r, g, b = rgb
    if max(rgb) < 45:
        return "black"
    if min(rgb) > 210:
        return "white"
    if abs(r - g) < 18 and abs(g - b) < 18:
        return "gray"
    if r > g and r > b:
        return "red_or_warm"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue_or_cool"
    return "mixed"


def local_image_analysis(image_path, product_title="", product_brand=""):
    path = Path(image_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        sample = image.resize((1, 1))
        dominant_rgb = sample.getpixel((0, 0))
        stat = ImageStat.Stat(image.resize((128, 128)))
        mean = stat.mean
        stddev = stat.stddev
        brightness = sum(mean) / 3
        contrast = sum(stddev) / 3
        colorfulness = max(mean) - min(mean)

    aspect_ratio = round(width / height, 3) if height else 0
    product_type = infer_product_type(product_title)
    dominant = dominant_color_name(dominant_rgb)
    visual_tags = [
        f"image_width_{width}",
        f"image_height_{height}",
        f"aspect_{bucket(aspect_ratio, 0.65, 1.45)}",
        f"brightness_{bucket(brightness, 85, 175)}",
        f"contrast_{bucket(contrast, 25, 70)}",
        f"dominant_color_{dominant}",
    ]
    risk_tags = ["not_enough_visual_evidence"]
    if product_type == "phone_accessory":
        risk_tags.append("possible_accessory_only")
    if contrast < 8:
        risk_tags.append("possible_blank_or_placeholder_image")

    caption_vi = (
        f"Ảnh sản phẩm cục bộ kích thước {width}x{height}, tông màu chủ đạo {dominant}, "
        f"độ sáng {bucket(brightness, 85, 175)}, độ tương phản {bucket(contrast, 25, 70)}."
    )
    caption_en = (
        f"Local product image features: {width}x{height}, dominant color {dominant}, "
        f"{bucket(brightness, 85, 175)} brightness, {bucket(contrast, 25, 70)} contrast."
    )
    return {
        "caption_vi": caption_vi,
        "caption_en": caption_en,
        "visible_product_type": product_type,
        "visible_brand_or_model": product_brand or None,
        "visual_tags": visual_tags,
        "condition_tags": ["unknown_condition"],
        "risk_tags": risk_tags,
        "evidence": [
            f"image_path: {path.name}",
            f"size: {width}x{height}",
            f"dominant_rgb: {dominant_rgb}",
            f"title_signal_product_type: {product_type}",
        ],
        "analysis_provider": "local_pillow",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/runtime/multimodal' / "image_analysis.jsonl")
    parser.add_argument("--limit", type=int, default=30, help="Maximum new images to analyze")
    parser.add_argument("--images-per-doc", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between API calls")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient Gemini errors")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry images that only have failed cache rows")
    parser.add_argument("--local-only", action="store_true",
                        help="Use local Pillow visual features without calling Gemini")
    parser.add_argument("--no-local-fallback", action="store_true",
                        help="Do not write local visual features if Gemini fails")
    parser.add_argument("--quiet", action="store_true", help="Only print final summary")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without calling Gemini")
    args = parser.parse_args()

    documents = read_jsonl(args.corpus.resolve())
    done = load_done_keys(args.output.resolve(), retry_errors=args.retry_errors)
    processed = 0
    candidates = 0

    for doc in documents:
        doc_id = doc.get("doc_id")
        for image_path in (doc.get("product_image_paths") or [])[:args.images_per_doc]:
            key = (doc_id, image_path)
            if key in done:
                continue
            if not Path(image_path).exists():
                continue
            candidates += 1
            if args.dry_run:
                print(json.dumps({
                    "doc_id": doc_id,
                    "title": doc.get("title"),
                    "brand": doc.get("brand"),
                    "image_path": image_path,
                }, ensure_ascii=False))
                processed += 1
            else:
                if not args.quiet:
                    print(json.dumps({
                        "status": "analyzing",
                        "doc_id": doc_id,
                        "title": doc.get("title"),
                        "image_path": image_path,
                        "mode": "local" if args.local_only else "gemini",
                    }, ensure_ascii=False), flush=True)
                if args.local_only:
                    result = {
                        "analysis": local_image_analysis(
                            image_path,
                            product_title=doc.get("title") or "",
                            product_brand=doc.get("brand") or "",
                        ),
                        "raw_text": None,
                        "error": None,
                        "model": "local_pillow",
                    }
                else:
                    result = generate_image_analysis(
                        image_path,
                        product_title=doc.get("title") or "",
                        product_brand=doc.get("brand") or "",
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    if result.get("analysis") is None and not args.no_local_fallback:
                        try:
                            fallback = local_image_analysis(
                                image_path,
                                product_title=doc.get("title") or "",
                                product_brand=doc.get("brand") or "",
                            )
                            fallback["vlm_error"] = result.get("error")
                            result = {
                                **result,
                                "analysis": fallback,
                                "error": None,
                                "fallback_error": result.get("error"),
                            }
                        except OSError as exc:
                            result["fallback_error"] = str(exc)
                append_jsonl(args.output.resolve(), {
                    "doc_id": doc_id,
                    "title": doc.get("title"),
                    "brand": doc.get("brand"),
                    "image_path": image_path,
                    "analysis": result.get("analysis"),
                    "raw_text": result.get("raw_text"),
                    "error": result.get("error"),
                    "fallback_error": result.get("fallback_error"),
                    "model": result.get("model"),
                    "created_at": int(time.time()),
                })
                if not args.quiet:
                    print(json.dumps({
                        "doc_id": doc_id,
                        "title": doc.get("title"),
                        "ok": result.get("analysis") is not None,
                        "error": result.get("error"),
                    }, ensure_ascii=False), flush=True)
                processed += 1
                if args.sleep:
                    time.sleep(args.sleep)
            if processed >= args.limit:
                print(json.dumps({
                    "status": "stopped_at_limit",
                    "new_items": processed,
                    "candidates_seen": candidates,
                    "output": str(args.output.resolve()),
                }, ensure_ascii=False, indent=2))
                return

    print(json.dumps({
        "status": "done",
        "new_items": processed,
        "candidates_seen": candidates,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
