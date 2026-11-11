"""Build a text-only RecSys split from MIND-small (user-sampled, leave-last-out on clicks).

Per user, ``history`` items (earlier clicks, order only) become training edges.  Clicks inside the
train period are ordered by impression time and de-duplicated; the user's last click is the test
target and the one before is the validation target, but only if that item is neither in the history
nor clicked earlier.  Everything else is training.  Users are sampled with a fixed seed before any
model is seen.  News items have only text (title, abstract, categories): no images or specifications,
so only text-based methods are meaningful.  The split is leave-last-out per user, not a global
chronological split: other users' clicks from the same days appear in training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from phone_assistant.data.preparation.prepare_amazon_category_recsys import popularity_report, write_jsonl
from phone_assistant.recommendation.recsys_data import ROOT, sha256

SOURCE = ROOT / "data/external/mind_small/MINDsmall_train"
OUTPUT = ROOT / "data/external/benchmarks/mind_small_text_v1"
SAMPLED_USERS = 6000
SEED = 20261110
MIN_CLICKS = 3  # distinct new clicks needed: test + validation + at least one training click
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
FEEDBACK_TYPE = "click"
HISTORY_TIMESTAMP_BASE = 0  # history order is kept by rank; real timestamps are unknown


def parse_time(text):
    return int(datetime.strptime(text, TIME_FORMAT).timestamp() * 1000)


def read_news(path):
    news = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            news_id, category, subcategory, title, abstract = columns[:5]
            news[news_id] = {"parent_asin": news_id, "title": title, "store": "",
                             "features": [abstract] if abstract.strip() else [], "description": [],
                             "categories": [category, subcategory], "details": {}, "images": []}
    return news


def read_behaviors(path):
    """user -> {"history": [...], "clicks": [(time_ms, news_id), ...]}; history from the user's earliest impression."""
    users = {}
    first_seen = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            _, user, stamp, history, impressions = line.rstrip("\n").split("\t")
            moment = parse_time(stamp)
            record = users.setdefault(user, {"history": [], "clicks": []})
            if user not in first_seen or moment < first_seen[user]:
                first_seen[user] = moment
                record["history"] = history.split() if history else []
            record["clicks"] += [(moment, item.rsplit("-", 1)[0]) for item in impressions.split() if item.endswith("-1")]
    return users


def user_events(record, known_items):
    """Return (training_items, validation_click, test_click) or None when the user has too few new clicks."""
    history = [item for item in dict.fromkeys(record["history"]) if item in known_items]
    seen = set(history)
    new_clicks = []
    for moment, item in sorted(record["clicks"]):
        if item in known_items and item not in seen:
            new_clicks.append((moment, item))
            seen.add(item)
    if len(new_clicks) < MIN_CLICKS:
        return None
    return history, new_clicks[:-2], new_clicks[-2], new_clicks[-1]


def build_split(users, known_items):
    """Rows per split; history items are ordered before any click so ties never reorder events."""
    split = {"train": [], "validation": [], "test": []}
    for user in sorted(users):
        events = user_events(users[user], known_items)
        if events is None:
            continue
        history, early, validation, test = events
        for rank, item in enumerate(history):
            split["train"].append({"user_id": user, "item_id": item, "timestamp": HISTORY_TIMESTAMP_BASE + rank,
                                   "rating": 1.0, "feedback_type": "history"})
        for moment, item in early:
            split["train"].append({"user_id": user, "item_id": item, "timestamp": moment, "rating": 1.0,
                                   "feedback_type": FEEDBACK_TYPE})
        for name, (moment, item) in (("validation", validation), ("test", test)):
            split[name].append({"user_id": user, "item_id": item, "timestamp": moment, "rating": 1.0,
                                "feedback_type": FEEDBACK_TYPE})
    return split


def prepare(source, output, sampled_users=SAMPLED_USERS, seed=SEED):
    if output.exists():
        raise FileExistsError("Preserve the benchmark; use a new output to revise it")
    news = read_news(source / "news.tsv")
    users = read_behaviors(source / "behaviors.tsv")
    eligible = sorted(user for user, record in users.items() if user_events(record, set(news)) is not None)
    chosen = set(random.Random(seed).sample(eligible, min(sampled_users, len(eligible))))
    split = build_split({user: users[user] for user in chosen}, set(news))
    user_ids = sorted({row["user_id"] for rows in split.values() for row in rows})
    item_ids = sorted({row["item_id"] for rows in split.values() for row in rows})
    output.mkdir(parents=True)
    write_jsonl(output / "products.jsonl", [news[item] for item in item_ids])
    for name, rows in split.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    write_jsonl(output / "train_graph_edges.jsonl", [
        {"source": row["user_id"], "target": row["item_id"], "relation": row["feedback_type"]} for row in split["train"]])
    (output / "id_maps.json").write_text(json.dumps({"users": {u: i for i, u in enumerate(user_ids)},
                                                     "items": {n: i for i, n in enumerate(item_ids)}}, indent=2), encoding="utf-8")
    (output / "popularity_metrics.json").write_text(json.dumps(popularity_report(
        split["train"], split["validation"], split["test"], item_ids), indent=2), encoding="utf-8")
    train_items = {row["item_id"] for row in split["train"]}
    summary = {"users": len(user_ids), "items": len(item_ids), "train": len(split["train"]),
               "train_history_edges": sum(row["feedback_type"] == "history" for row in split["train"]),
               "train_click_edges": sum(row["feedback_type"] == FEEDBACK_TYPE for row in split["train"]),
               "validation": len(split["validation"]), "test": len(split["test"]),
               "test_targets_without_train_edge": sum(row["item_id"] not in train_items for row in split["test"]),
               "eligible_users_before_sampling": len(eligible), "users_in_source": len(users)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts = ("products.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl", "train_graph_edges.jsonl",
                 "id_maps.json", "popularity_metrics.json", "summary.json")
    manifest = {"schema_version": 1, "dataset": "MIND-small train period (news recommendation)",
                "source_sha256": {"news": sha256(source / "news.tsv"), "behaviors": sha256(source / "behaviors.tsv"),
                                  "preparer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "config": {"sampled_users": sampled_users, "seed": seed, "min_new_clicks": MIN_CLICKS,
                           "split": "per-user leave-last-out on new clicks by impression time; history is training only",
                           "history_timestamps": "unknown; kept in order with rank as pseudo time before any click"},
                "artifacts_sha256": {name: sha256(output / name) for name in artifacts},
                "limitations": ["User-sampled subset of MIND-small; not the official impression-based evaluation.",
                                "Per-user leave-last-out, not a global chronological split.",
                                "Text only: no images or technical specifications; only text-based methods apply.",
                                "News is short-lived, so many held-out items have few training edges (cold targets are kept and reported)."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sampled-users", type=int, default=SAMPLED_USERS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.sampled_users, args.seed), indent=2))


if __name__ == "__main__":
    main()
