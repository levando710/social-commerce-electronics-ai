"""Separate, train-only public opinions; never hardware truth or a browser user's history."""
import argparse
import hashlib
import json
from pathlib import Path

from phone_assistant.retrieval.search.bm25 import build_index, score
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, read_jsonl, sha256

STORE = ROOT / 'artifacts/cache/handset_review_evidence_v1'


def select_reviews(data, reviews):
    allowed = {(r["user_id"], r["item_id"], int(r["timestamp"])) for r in data.rows["train"]}
    heldout = {(r["user_id"], r["item_id"]) for split in ("validation", "test") for r in data.rows[split]}
    documents = {}
    for review in reviews:
        user = hashlib.sha256(str(review.get("user_id", "")).encode()).hexdigest()
        item = review.get("parent_asin")
        timestamp = int(review.get("timestamp", 0))
        if (user, item, timestamp) not in allowed or (user, item) in heldout:
            continue
        text = str(review.get("text") or "").strip()
        if not text:
            continue
        title = str(review.get("title") or "").strip()
        identity = hashlib.sha256(json.dumps([user, item, timestamp, title, text], ensure_ascii=False).encode()).hexdigest()
        documents[identity] = {"doc_id": "review_" + identity[:24], "item_id": item, "title": title,
            "text": text, "timestamp": timestamp, "source_dataset": str(review.get("source_dataset") or ""),
            "evidence_type": "public_train_review_opinion", "hardware_verified": False}
    return sorted(documents.values(), key=lambda r: r["doc_id"])


class ReviewEvidence:
    def __init__(self, path=STORE):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["split_manifest_sha256"] != sha256(ROOT / 'data/external/benchmarks/handsets_only_v2/manifest.json'):
            raise ValueError("Review evidence split changed")
        for name, digest in manifest["artifacts_sha256"].items():
            if sha256(path / name) != digest:
                raise ValueError("Review evidence artifact changed")
        self.documents = {d["doc_id"]: d for d in read_jsonl(path / "documents.jsonl")}
        self.index = json.loads((path / "index.json").read_text(encoding="utf-8"))

    def search(self, item, query="", top_k=3):
        rows = [d for d in self.documents.values() if d["item_id"] == item]
        if query.strip():
            scores = dict(score(self.index, query))
            rows = [r for r in rows if scores.get(r["doc_id"], 0) > 0]
            rows.sort(key=lambda r: (-scores[r["doc_id"]], r["doc_id"]))
        else:
            rows.sort(key=lambda r: (-r["timestamp"], r["doc_id"]))
        return rows[:max(0, min(top_k, 10))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STORE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new version; retain frozen review evidence")
    data = InteractionData(ROOT / 'data/external/benchmarks/handsets_only_v2')
    source = ROOT / 'data/collections/phones_filtered/processed/reviews.jsonl'
    rows = select_reviews(data, read_jsonl(source))
    args.output.mkdir(parents=True)
    (args.output / "documents.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (args.output / "index.json").write_text(json.dumps(build_index(rows)), encoding="utf-8")
    manifest = {"split_manifest_sha256": sha256(data.directory / "manifest.json"), "source_reviews_sha256": sha256(source),
        "source_sha256": sha256(Path(__file__)), "reviews": len(rows), "items": len({r["item_id"] for r in rows}),
        "api_calls": 0, "policy": "Exact hashed user+item+timestamp match to frozen train; exclude heldout pairs; no exported user IDs.",
        "limitations": "Unverified historical opinions, selected positive train interactions only; not representative sentiment, hard specs, current device condition or the visitor's history.",
        "artifacts_sha256": {n: sha256(args.output / n) for n in ("documents.jsonl", "index.json")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
