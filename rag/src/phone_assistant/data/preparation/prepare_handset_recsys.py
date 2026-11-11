"""Build a new handset-only metadata catalog and chronological review benchmark.

The catalog is selected before seeing feedback. Historical simulated-query users
are excluded from the review archive; no conversations or new labels are invented.
"""
from phone_assistant.common.project_layout import source_path
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from phone_assistant.common.catalog_schema import catalog_text
from phone_assistant.data.phone_scope import handset_decision
from phone_assistant.data.preparation.prepare_connected_recsys import connect_cache, import_catalog_rows, choose_support
from phone_assistant.data.preparation.prepare_recsys import write_jsonl, temporal_split, k_core_size, popularity_metrics
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, read_jsonl, sha256

PRODUCT_KEYS = ("parent_asin", "title", "features", "description", "categories", "details", "store", "images")


def select_catalog(sources):
    selected, provenance = {}, {}
    for source_name, products, documents in sources:
        by_id = {d["doc_id"]: d for d in documents}
        for original in sorted(products, key=lambda p: p["parent_asin"]):
            item = original["parent_asin"]
            if item in selected or not handset_decision(original)[0]:
                continue
            product = {key: original.get(key) for key in PRODUCT_KEYS}
            product["details"] = original.get("details") or {}
            selected[item] = (product, by_id.get(item, {}))
            provenance[item] = source_name
    return selected, provenance


def split_with_context(interactions):
    groups = defaultdict(list)
    for row in interactions:
        groups[row["user_id"]].append(row)
    eligible = [r for rows in groups.values() if len(rows) >= 3 for r in rows]
    context = [r for rows in groups.values() if len(rows) < 3 for r in rows]
    train, validation, test, tied = temporal_split(eligible)
    train += context
    return {name: sorted(rows, key=lambda r: (r["user_id"], r["timestamp"], r["item_id"]))
            for name, rows in (("train", train), ("validation", validation), ("test", test))}, tied


def export_split(directory, catalog, provenance, interactions, contract):
    splits, tied = split_with_context(interactions)
    users = sorted({r["user_id"] for r in splits["train"]})
    items = sorted(catalog)
    documents = []
    for item in items:
        product, original_doc = catalog[item]
        documents.append({"doc_id": item, "parent_asin": item, "title": product["title"],
            "details": product["details"], "categories": product.get("categories") or [],
            "brand": product["details"].get("Brand") or product["details"].get("Manufacturer"),
            "text": catalog_text(product), "reviews": [], "visual_evidence": [], "review_image_paths": [],
            "product_image_paths": [p for p in original_doc.get("product_image_paths", []) if Path(p).is_file()]})
    write_jsonl(directory / "products.jsonl", [catalog[item][0] for item in items])
    write_jsonl(directory / "documents.jsonl", documents)
    for name, rows in splits.items():
        write_jsonl(directory / f"{name}.jsonl", rows)
    write_jsonl(directory / "train_graph_edges.jsonl", [{"source": r["user_id"], "target": r["item_id"], "relation": "positive_review"} for r in splits["train"]])
    (directory / "id_maps.json").write_text(json.dumps({"users": {u: i for i, u in enumerate(users)}, "items": {item: i for i, item in enumerate(items)}}, indent=2), encoding="utf-8")
    (directory / "user_roles.json").write_text(json.dumps({u: "handset_review_history" for u in users}, indent=2), encoding="utf-8")
    (directory / "catalog_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    pop = {"method": "Pop", "validation": popularity_metrics(splits["train"], splits["validation"], items),
        "test": popularity_metrics(splits["train"], splits["test"], items, exclude=splits["validation"])}
    (directory / "popularity_metrics.json").write_text(json.dumps(pop, indent=2), encoding="utf-8")
    degree = Counter(r["item_id"] for r in splits["train"])
    summary = {"scope": "handsets_only_automatic_metadata_screening", "users": len(users), "items": len(items),
        "catalog_sources": dict(Counter(provenance.values())), "split_events": {n: len(r) for n, r in splits.items()},
        "train_two_core": k_core_size(splits["train"]), "test_cold_targets": sum(not degree[r["item_id"]] for r in splits["test"]),
        "timestamp_tied_users_train_only": tied, "items_with_train_feedback": len(degree),
        "images_available": sum(bool(d["product_image_paths"]) for d in documents), "archive_audit": contract.pop("archive_audit"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Automatic conservative handset labels need human review; not a manually verified catalog.",
            "Reviews proxy positive feedback, not actual purchases or clicks.", "Per-user chronological holdout, not a global time cutoff.",
            "Evaluation users require at least three distinct positive handset reviews: activity-conditioned cohort. Other users with one/two positives are train-only context.",
            "Catalog metadata is static and transductive; not a historical time-versioned store.",
            "No conversational supervision is created; old mixed-catalog dialogue metrics are not reused.",
            "This split is new but the source data and old model results have previously been inspected; no claim of an untouched external benchmark."]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, **contract, "config": {"positive_rating": 4., "min_eval_positive_items": 3, "low_degree_policy": "train_only_context",
        "protocol": "per-user chronological leave-last-two-out; strict timestamps; full handset catalog"},
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix != ".sqlite"}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frozen = InteractionData(directory)
    if any(not handset_decision(p)[0] for p, _ in catalog.values()):
        raise ValueError("Non-handset leaked into the catalog")
    for name in ("validation", "test"):
        for row in frozen.rows[name]:
            earlier = [r["timestamp"] for r in frozen.rows["train"] if r["user_id"] == row["user_id"]]
            if not earlier or max(earlier) >= row["timestamp"]:
                raise ValueError("Holdout is not strictly later than train")
    return summary, pop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/benchmarks/handsets_only_v2')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep frozen artifacts; use a new output")
    paths = {"legacy_products": ROOT / 'data/collections/phones_filtered/processed/products.jsonl',
        "legacy_documents": ROOT / 'artifacts/runtime/catalog/documents.jsonl',
        "connected_products": ROOT / 'data/external/benchmarks/connected_phones_v1/products.jsonl',
        "connected_documents": ROOT / 'artifacts/cache/connected_runtime_v1/documents.jsonl',
        "archive": ROOT / 'data/external/amazon_5core/Cell_Phones_and_Accessories.csv.gz',
        "archive_manifest": ROOT / 'data/external/amazon_5core/manifest.json',
        "protected_users": ROOT / 'data/external/benchmarks/phones_v1/id_maps.json'}
    archive_manifest = json.loads(paths["archive_manifest"].read_text(encoding="utf-8"))
    if sha256(paths["archive"]) != archive_manifest["sha256"]:
        raise ValueError("Source archive changed")
    catalog, provenance = select_catalog([(name, read_jsonl(paths[f"{name}_products"]), read_jsonl(paths[f"{name}_documents"]))
        for name in ("legacy", "connected")])
    protected = set(json.loads(paths["protected_users"].read_text(encoding="utf-8"))["users"])
    args.output.mkdir(parents=True)
    connection = connect_cache(args.output / "selection.sqlite")
    try:
        with gzip.open(paths["archive"], "rt", encoding="utf-8-sig", newline="") as handle:
            audit = import_catalog_rows(connection, csv.DictReader(handle), set(catalog), protected)
        interactions, user_ids = choose_support(connection, threshold=4., min_items=1, max_users=1000000)
    finally:
        connection.close()
    if not interactions:
        raise ValueError("No eligible handset histories; do not fabricate interactions")
    contract = {"source_sha256": {name: sha256(p) for name, p in paths.items()},
        "preparer_sha256": {name: sha256(source_path(name)) for name in ("prepare_handset_recsys.py", "phone_scope.py", "product_constraints.py", "prepare_connected_recsys.py", "prepare_recsys.py", "catalog_schema.py")},
        "excluded_historical_query_users": len(protected), "archive_audit": audit}
    summary, pop = export_split(args.output, catalog, provenance, interactions, contract)
    print(json.dumps({"summary": summary, "popularity": pop}, indent=2), flush=True)


if __name__ == "__main__":
    main()
