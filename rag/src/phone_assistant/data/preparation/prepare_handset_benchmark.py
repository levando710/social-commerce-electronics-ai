"""Freeze a bilingual catalog-criteria diagnostic; not observed dialogue or train supervision."""
import json
from pathlib import Path
import re

from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def gold_fields(product):
    details = product.get("details") or {}
    raw = str(details.get("Brand") or details.get("Manufacturer") or "").lower().strip()
    brand = next((b for b in ("samsung", "apple", "motorola", "nokia", "google", "lg") if raw == b or raw.startswith(b + " ")), None)
    values = {"brand": brand}
    for field, keys in (("storage_gb", ("Memory Storage Capacity", "Storage Capacity")), ("ram_gb", ("RAM", "RAM Memory Installed Size", "Memory RAM"))):
        numbers = []
        for key in keys:
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*(GB|TB|MB)\b", str(details.get(key, "")), re.I)
            numbers += [float(n) * (1024 if u.upper() == "TB" else 1 / 1024 if u.upper() == "MB" else 1) for n, u in matches]
        values[field] = numbers[0] if numbers and len(set(numbers)) == 1 else None
    carrier = str(details.get("Wireless Carrier") or "").lower()
    values["network"] = "unlocked" if "unlocked" in carrier else "locked" if "locked" in carrier else None
    raw_battery = str(details.get("Battery Capacity") or "")
    first = re.match(r"([\d,]+(?:\.\d+)?)\s+(?:Milliamp Hours?|mAh)\b", raw_battery, re.I)
    values["battery_min_mah"] = float(first.group(1).replace(",", "")) if first else None
    return values


def judge_fields(fields, constraints):
    unknown = False
    for key, expected in constraints.items():
        observed = fields.get(key)
        if observed is None:
            unknown = True
        elif (key == "battery_min_mah" and observed < expected) or (key != "battery_min_mah" and observed != expected):
            return "irrelevant"
    return "unjudged" if unknown else "relevant"


def main():
    output = ROOT / 'artifacts/evaluation/handset_benchmark_v1'
    if output.exists():
        raise FileExistsError("Keep the frozen benchmark")
    split = ROOT / 'data/external/benchmarks/handsets_only_v2'
    products = read_jsonl(split / "products.jsonl")
    intents = [
        ("development", "samsung", {"brand": "samsung"}, "điện thoại Samsung", "Samsung phone"),
        ("development", "apple", {"brand": "apple"}, "điện thoại Apple iPhone", "Apple iPhone phone"),
        ("development", "samsung128", {"brand": "samsung", "storage_gb": 128}, "điện thoại Samsung 128GB", "Samsung phone with 128GB storage"),
        ("development", "samsung64", {"brand": "samsung", "storage_gb": 64}, "điện thoại Samsung 64GB", "Samsung phone with 64GB storage"),
        ("development", "apple128", {"brand": "apple", "storage_gb": 128}, "iPhone 128GB", "Apple iPhone 128GB storage"),
        ("development", "ram8", {"ram_gb": 8, "storage_gb": 128}, "điện thoại 8GB RAM 128GB", "phone 8GB RAM and 128GB storage"),
        ("development", "unlocked64", {"network": "unlocked", "storage_gb": 64}, "điện thoại 64GB không khóa mạng", "unlocked phone 64GB storage"),
        ("development", "battery4000", {"battery_min_mah": 4000}, "điện thoại pin ít nhất 4000 mAh", "phone with at least 4000 mAh battery"),
        ("test", "motorola", {"brand": "motorola"}, "điện thoại Motorola", "Motorola phone"),
        ("test", "nokia", {"brand": "nokia"}, "điện thoại Nokia", "Nokia phone"),
        ("test", "google", {"brand": "google"}, "điện thoại Google Pixel", "Google Pixel phone"),
        ("test", "samsung256", {"brand": "samsung", "storage_gb": 256}, "điện thoại Samsung 256GB", "Samsung phone 256GB storage"),
        ("test", "apple64", {"brand": "apple", "storage_gb": 64}, "iPhone 64GB", "Apple iPhone phone 64GB storage"),
        ("test", "ram4", {"ram_gb": 4, "storage_gb": 64}, "điện thoại 4GB RAM 64GB", "phone 4GB RAM 64GB storage"),
        ("test", "unlocked128", {"network": "unlocked", "storage_gb": 128}, "điện thoại 128GB không khóa mạng", "unlocked phone 128GB storage"),
        ("test", "lg32", {"brand": "lg", "storage_gb": 32}, "điện thoại LG 32GB", "LG phone 32GB storage"),
    ]
    queries, qrels = [], {}
    for cohort, intent, constraints, vietnamese, english in intents:
        labels = {p["parent_asin"]: judge_fields(gold_fields(p), constraints) for p in products}
        if "relevant" not in labels.values():
            raise ValueError(f"No explicit-metadata relevant products for {intent}; benchmark not frozen")
        qrels[intent] = labels
        for language, phrase in (("vi", vietnamese), ("en", english)):
            for variant in (0, 1):
                prefix = ("Tìm giúp tôi " if language == "vi" else "Please find ") if variant else ""
                queries.append({"id": f"{intent}_{language}_{variant}", "intent_id": intent, "cohort": cohort,
                    "language": language, "query": prefix + phrase, "constraints": constraints, "kind": "retrieval"})
    for cohort, intent, texts in (("development", "price_unknown", ("điện thoại Samsung dưới 5 triệu", "Samsung phone under 300 USD")),
        ("development", "case_outside", ("ốp lưng cho iPhone 13", "case for iPhone 13")),
        ("test", "tablet_outside", ("tìm máy tính bảng iPad", "find an iPad tablet")),
        ("test", "charger_outside", ("tìm củ sạc cho Samsung", "charger for Samsung"))):
        for language, query in zip(("vi", "en"), texts):
            queries.append({"id": f"{intent}_{language}", "intent_id": intent, "cohort": cohort, "language": language,
                "query": query, "constraints": {}, "kind": "expected_abstention"})
    output.mkdir(parents=True)
    (output / "queries.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    (output / "qrels.json").write_text(json.dumps(qrels, indent=2), encoding="utf-8")
    manifest = {"split_manifest_sha256": sha256(split / "manifest.json"), "source_sha256": sha256(Path(__file__)),
        "queries": len(queries), "retrieval_intents": len(intents), "cohort_policy": "8 development and 8 test criteria; paraphrases/languages stay in one cohort",
        "label_policy": "Independent raw structured metadata criteria, binary relevant/irrelevant and unjudged for missing fields; not human relevance labels.",
        "scope": "Researcher-authored catalog-template diagnostic, not user dialogs, joint-training examples or an external untouched benchmark.",
        "artifacts_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"queries": len(queries), "qrel_relevant_counts": {k: list(v.values()).count('relevant') for k, v in qrels.items()}}, indent=2))


if __name__ == "__main__":
    main()
