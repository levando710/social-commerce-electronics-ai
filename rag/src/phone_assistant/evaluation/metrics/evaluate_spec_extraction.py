"""Does extra evidence make attribute extraction more accurate, or more prone to wrong answers?

Gold values are the seller-entered structured ``details`` of Amazon Electronics products.  Those fields are
removed from every input: the model only sees the listing text (title, store, bullets, description,
categories), optionally the text of other retrieved listings (which may be different products) and the
catalog photo.  An answer is correct, wrong (a hallucinated or misread value) or absent.  The study is a
fixed protocol: a small development split is only for debugging the pipeline; the prompt, quotas and
conditions are frozen in ``protocol.json`` before the test split is run once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import partial
from pathlib import Path

import numpy as np

from phone_assistant.common.image_mime import image_mime
from phone_assistant.evaluation.metrics.run_generation_expansion import (
    OPENAI_PRICES, OPENAI_PRICING_DATE, RequestLedger, append_jsonl, verified_image_parts, worst_case_cost_usd, write_json)
from phone_assistant.llm import openai_client
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256
from phone_assistant.retrieval.search.bm25 import build_index, score

SPLIT_DIRECTORY = ROOT / "data/external/benchmarks/connected_electronics_v2"
IMAGE_DIRECTORY = ROOT / "artifacts/cache/electronics_catalog_work_v1/images"
OUTPUT_ROOT = ROOT / "artifacts/evaluation"
ATTRIBUTES = ("color", "brand", "screen_inch", "os", "storage_gb", "ram_gb")
QUOTAS = {"color": 100, "brand": 60, "screen_inch": 100, "os": 80, "storage_gb": 80, "ram_gb": 60}
CONDITIONS = {"text": (False, False), "text_neighbors": (True, False), "text_image": (False, True),
              "text_neighbors_image": (True, True)}
SEED = 20261110
DEV_FRACTION = 0.2
NEIGHBORS = 3
NEIGHBOR_CHARS = 700
OWN_TEXT_CHARS = 3000
OUTPUT_TOKENS = 400
MODEL = "gpt-4.1-mini"
BOOTSTRAP_REPETITIONS = 1000
INSTRUCTION = (
    "You extract attributes of ONE consumer-electronics product listing. Use ONLY the evidence below: the listing text "
    "(source own_text); optionally other listings (source neighbor; they are DIFFERENT products and may have different "
    "specifications); optionally product photos (source image). Never guess. If the evidence does not state an attribute "
    "for THIS product, use null. Return a JSON object with exactly these keys: color, brand, screen_inch, os, "
    "storage_gb, ram_gb. Each value is null or an object {\"value\": ..., \"source\": \"own_text\"|\"neighbor\"|\"image\", "
    "\"quote\": \"exact text copied from the evidence (empty for image)\"}. Units: screen_inch a number in inches; "
    "storage_gb and ram_gb numbers in GB (convert TB to GB); color the main color name; os the operating system name; "
    "brand the manufacturer brand.\n")


# --------------------------------------------------------------------------- gold and evidence

def to_gb(value):
    match = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB|MB)\b", str(value), re.I)
    if not match:
        return None
    number, unit = float(match.group(1)), match.group(2).upper()
    return number * 1024 if unit == "TB" else number if unit == "GB" else number / 1024


def to_inches(value):
    match = re.search(r"(\d+(?:\.\d+)?)\s*inch", str(value), re.I)
    return float(match.group(1)) if match else None


def gold_attributes(product):
    """Structured seller fields only; never part of any model input."""
    details = product.get("details") or {}
    gold = {}
    if str(details.get("Color") or "").strip():
        gold["color"] = str(details["Color"])
    if str(details.get("Brand") or "").strip():
        gold["brand"] = str(details["Brand"])
    for key in ("Screen Size", "Standing screen display size", "Display Size"):
        if details.get(key) and to_inches(details[key]):
            gold["screen_inch"] = to_inches(details[key])
            break
    if str(details.get("Operating System") or "").strip():
        gold["os"] = str(details["Operating System"])
    for key in ("Memory Storage Capacity", "Hard Drive", "Digital Storage Capacity"):
        if details.get(key) and to_gb(details[key]):
            gold["storage_gb"] = to_gb(details[key])
            break
    if details.get("RAM") and to_gb(details["RAM"]):
        gold["ram_gb"] = to_gb(details["RAM"])
    return gold


def evidence_text(product, limit=OWN_TEXT_CHARS):
    """Listing text without ``details`` and without anything derived from it (``catalog_text`` adds spec tokens)."""
    parts = [str(product.get("title") or ""), str(product.get("store") or "")]
    for key in ("features", "description", "categories"):
        values = product.get(key) or []
        parts.extend([values] if isinstance(values, str) else map(str, values))
    return re.sub(r"\s+", " ", " \n".join(part for part in parts if part.strip())).strip()[:limit]


def item_hash(asin, salt):
    return hashlib.sha256(f"{salt}:{asin}".encode()).hexdigest()


def select_items(products, quotas=QUOTAS, seed=SEED, dev_fraction=DEV_FRACTION):
    """Deterministic quota sample per attribute, then a hash-based dev/test split of the union."""
    chosen = set()
    for attribute, quota in quotas.items():
        pool = sorted((p["parent_asin"] for p in products if attribute in gold_attributes(p)),
                      key=lambda asin: item_hash(asin, f"{seed}:{attribute}"))
        chosen.update(pool[:quota])
    dev = {asin for asin in chosen if int(item_hash(asin, f"{seed}:split")[:8], 16) / 0xFFFFFFFF < dev_fraction}
    return {"dev": sorted(dev), "test": sorted(chosen - dev)}


class NeighborFinder:
    def __init__(self, products):
        self.texts = {p["parent_asin"]: evidence_text(p) for p in products}
        self.titles = {p["parent_asin"]: str(p.get("title") or "") for p in products}
        self.index = build_index([{"doc_id": asin, "text": text} for asin, text in self.texts.items()])

    def find(self, asin, count=NEIGHBORS):
        ranked = [doc_id for doc_id, _ in score(self.index, self.titles[asin])[:count + 1] if doc_id != asin]
        return ranked[:count]


def image_entry(asin, directory=None):
    directory = directory or IMAGE_DIRECTORY
    record = json.loads((directory / f"{asin}.json").read_text(encoding="utf-8"))
    path = directory / f"{asin}.img"
    if record.get("status") != "ok" or sha256(path) != record["sha256"]:
        return None
    return {"image_id": "image_1", "doc_id": asin, "path": str(path), "sha256": record["sha256"], "mime_type": image_mime(path)}


def build_request(product, neighbors, use_neighbors, use_image, finder, image):
    own = evidence_text(product)
    text = INSTRUCTION + "\nSOURCE own_text:\n" + own + "\n"
    neighbor_texts = {}
    if use_neighbors:
        for neighbor in neighbors:
            neighbor_texts[neighbor] = finder.texts[neighbor][:NEIGHBOR_CHARS]
            text += f"\nSOURCE neighbor ({neighbor}, a different listing):\n{neighbor_texts[neighbor]}\n"
    parts = [{"text": text}]
    if use_image and image is not None:
        parts += verified_image_parts([image])
    return parts, own, neighbor_texts


# --------------------------------------------------------------------------- parsing and matching

STOP = {"and", "with", "color", "colour", "the"}
COLOR_SYNONYMS = {"aluminum": "silver", "aluminium": "silver", "metallic": "silver", "steel": "silver", "platinum": "silver",
                  "gunmetal": "gray", "graphite": "gray", "charcoal": "gray", "slate": "gray", "grey": "gray",
                  "jet": "black", "midnight": "black", "ivory": "white", "cream": "white"}
MATCHING_RULES = "v2: GB/TB tolerate 1000- and 1024-based units (1%); color synonyms; invalid gold excluded at scoring"
OS_FAMILIES = ("windows", "android", "ios", "macos", "mac", "linux", "chrome", "fire", "tizen", "webos", "roku")


def words(value):
    return {w for w in re.findall(r"[a-z]+", str(value).lower()) if w not in STOP}


def color_words(value):
    return {COLOR_SYNONYMS.get(w, w) for w in words(value)}


def valid_gold(attribute, gold):
    """Seller fields are noisy: a 'color' such as '4 channel 960H DVR Kit' is not a color and is not scored."""
    text = str(gold)
    if attribute == "color":
        return len(text.split()) <= 3 and not re.search(r"\d", text)
    if attribute == "brand":
        return len(text.split()) <= 4
    return True


def numeric(value):
    try:
        return float(re.search(r"\d+(?:\.\d+)?", str(value)).group())
    except (AttributeError, ValueError):
        return None


def matches(attribute, gold, answer):
    if attribute in ("storage_gb", "ram_gb"):
        value = numeric(answer)
        return value is not None and any(abs(value - g) <= max(0.01 * g, 0.001)
                                         for g in (gold, gold * 1024 / 1000, gold * 1000 / 1024))
    if attribute == "screen_inch":
        value = numeric(answer)
        return value is not None and abs(value - gold) <= 0.15
    if attribute == "color":
        return bool(color_words(gold) & color_words(answer))
    if attribute == "brand":
        a, b = re.sub(r"[^a-z0-9]", "", str(gold).lower()), re.sub(r"[^a-z0-9]", "", str(answer).lower())
        return bool(a and b and (a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a))))
    families = {f for f in OS_FAMILIES if f in str(gold).lower()}
    return bool(families & {f for f in OS_FAMILIES if f in str(answer).lower()}) or words(gold) == words(answer)


def gold_in_text(attribute, gold, text):
    """Is the gold value stated literally in the item's own listing text?"""
    lowered = text.lower()
    if attribute in ("storage_gb", "ram_gb"):
        return any(abs(g - gold) <= 0.01 * gold for g in (to_gb(m.group()) for m in
                   re.finditer(r"\d+(?:\.\d+)?\s*(?:TB|GB|MB)\b", text, re.I)) if g)
    if attribute == "screen_inch":
        return any(abs(float(m.group(1)) - gold) <= 0.15 for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:inch|in\b|\")", lowered))
    if attribute == "os":
        return any(f in lowered for f in OS_FAMILIES if f in str(gold).lower())
    return any(w in lowered for w in words(gold)) if words(gold) else False


def parse_prediction(text, own, neighbor_texts):
    """-> {attribute: {"value","source","grounded"} | None}; ``grounded`` checks the quote against the given text."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    parsed = {}
    for attribute in ATTRIBUTES:
        entry = value.get(attribute)
        if entry is None:
            parsed[attribute] = None
            continue
        if not isinstance(entry, dict) or entry.get("value") in (None, ""):
            parsed[attribute] = None
            continue
        source, quote = entry.get("source"), str(entry.get("quote") or "")
        haystack = own if source == "own_text" else " ".join(neighbor_texts.values()) if source == "neighbor" else None
        grounded = None if haystack is None else bool(quote) and quote.lower() in haystack.lower()
        parsed[attribute] = {"value": entry["value"], "source": source, "grounded": grounded}
    return parsed


# --------------------------------------------------------------------------- rule baseline

COLORS = ("black", "white", "silver", "gray", "grey", "blue", "red", "green", "gold", "pink", "purple", "yellow",
          "orange", "brown", "rose gold", "space gray", "midnight", "graphite", "beige", "clear")


def rule_baseline(product):
    """Crude regular expressions on the listing text only; it exists to show what extra evidence must beat."""
    text = evidence_text(product)
    lowered = text.lower()
    found = {}
    ram = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)\s*(?:of\s*)?(?:DDR\d*\s*)?(?:RAM|memory)\b", text, re.I) or \
        re.search(r"\bRAM\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(TB|GB)", text, re.I)
    if ram:
        found["ram_gb"] = to_gb(ram.group(1) + " " + ram.group(2))
    storage = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)\s*(?:SSD|HDD|eMMC|ROM|flash|storage|hard drive|internal)", text, re.I)
    if storage:
        found["storage_gb"] = to_gb(storage.group(1) + " " + storage.group(2))
    screen = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:inch|in\b|\")", lowered)
    if screen:
        found["screen_inch"] = float(screen.group(1))
    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", str(product.get("title") or "").lower()):
            found["color"] = color
            break
    for family in OS_FAMILIES:
        if re.search(rf"\b{family}\b", lowered):
            found["os"] = family
            break
    title_words = str(product.get("title") or "").split()
    if title_words:
        found["brand"] = title_words[0]
    return {attribute: ({"value": found[attribute], "source": "own_text", "grounded": True} if attribute in found else None)
            for attribute in ATTRIBUTES}


# --------------------------------------------------------------------------- scoring

def score_records(records, products):
    """records: [{asin, condition, prediction}] -> per attribute x condition counts and rates."""
    by_asin = {p["parent_asin"]: p for p in products}
    cells = {}
    for record in records:
        product = by_asin[record["asin"]]
        own = evidence_text(product)
        for attribute, gold in gold_attributes(product).items():
            if attribute not in ATTRIBUTES or not valid_gold(attribute, gold):
                continue
            answer = (record["prediction"] or {}).get(attribute)
            literal = gold_in_text(attribute, gold, own)
            outcome = "absent" if answer is None else "correct" if matches(attribute, gold, answer["value"]) else "wrong"
            cells.setdefault((attribute, record["condition"]), []).append(
                {"asin": record["asin"], "outcome": outcome, "in_text": literal,
                 "source": (answer or {}).get("source"), "grounded": (answer or {}).get("grounded")})
    return cells


def rates(rows):
    n = len(rows)
    correct = sum(r["outcome"] == "correct" for r in rows)
    wrong = sum(r["outcome"] == "wrong" for r in rows)
    answered = correct + wrong
    return {"n": n, "correct": correct, "wrong": wrong, "absent": n - answered,
            "recall": correct / n if n else None, "precision": correct / answered if answered else None,
            "wrong_rate": wrong / n if n else None}


def bootstrap(rows, key, seed=SEED, repetitions=BOOTSTRAP_REPETITIONS):
    if len(rows) < 2:
        return [None, None]
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = [rows[i] for i in generator.integers(len(rows), size=len(rows))]
        value = rates(sample)[key]
        if value is not None:
            values.append(value)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [None, None]


def summarize(cells):
    summary = {}
    for (attribute, condition), rows in sorted(cells.items()):
        summary.setdefault(attribute, {})[condition] = {
            **rates(rows), "recall_ci95": bootstrap(rows, "recall"), "wrong_rate_ci95": bootstrap(rows, "wrong_rate"),
            "when_gold_not_in_own_text": rates([r for r in rows if not r["in_text"]]),
            "when_gold_in_own_text": rates([r for r in rows if r["in_text"]]),
            "answers_from_neighbor": sum(r["source"] == "neighbor" for r in rows),
            "answers_from_image": sum(r["source"] == "image" for r in rows),
            "ungrounded_quotes": sum(r["grounded"] is False for r in rows)}
    return summary


PAIRED_CONTRASTS = (("text", "text_neighbors"), ("text", "text_image"), ("text", "text_neighbors_image"),
                    ("text_image", "text_neighbors_image"))


def paired_differences(cells, repetitions=BOOTSTRAP_REPETITIONS, seed=SEED):
    """second - first on the same items: change in recall (correct) and in wrong rate, with a bootstrap over items."""
    result = {}
    for attribute in ATTRIBUTES:
        for first, second in PAIRED_CONTRASTS:
            a = {r["asin"]: r["outcome"] for r in cells.get((attribute, first), [])}
            b = {r["asin"]: r["outcome"] for r in cells.get((attribute, second), [])}
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                continue
            generator = np.random.default_rng(seed)
            entry = {"items": len(common)}
            for name, target in (("recall_delta", "correct"), ("wrong_rate_delta", "wrong")):
                deltas = np.array([float(b[i] == target) - float(a[i] == target) for i in common])
                samples = [float(deltas[generator.integers(len(common), size=len(common))].mean()) for _ in range(repetitions)]
                entry[name] = {"mean": float(deltas.mean()), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}
            result.setdefault(attribute, {})[f"{second}_minus_{first}"] = entry
    return result


# --------------------------------------------------------------------------- run

def protocol_document(selection):
    return {"model": MODEL, "attributes": list(ATTRIBUTES), "quotas": QUOTAS, "conditions": list(CONDITIONS),
            "seed": SEED, "dev_fraction": DEV_FRACTION, "neighbors": NEIGHBORS, "neighbor_chars": NEIGHBOR_CHARS,
            "own_text_chars": OWN_TEXT_CHARS, "output_tokens": OUTPUT_TOKENS, "temperature": 0,
            "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
            "items": {name: len(items) for name, items in selection.items()},
            "selection_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
            "gold_policy": "seller-entered details fields; never shown to the model; leniency rules in matches()",
            "matching_rules": MATCHING_RULES,
            "pricing": {"input_usd_per_million": OPENAI_PRICES[MODEL][0], "output_usd_per_million": OPENAI_PRICES[MODEL][1],
                        "pricing_checked_date": OPENAI_PRICING_DATE}}


def run(split, output, products, *, budget_usd, max_requests, live, caller, min_gap_seconds=0.3, resume_after_stop=False):
    selection = select_items(products)
    protocol = protocol_document(selection)
    by_asin = {p["parent_asin"]: p for p in products}
    finder = NeighborFinder(products)
    items = selection[split]
    jobs = [(asin, condition) for asin in items for condition in CONDITIONS]
    if not live:
        bound = sum(worst_case_cost_usd(build_request(by_asin[a], finder.find(a), *CONDITIONS[c], finder, image_entry(a))[0],
                                        OUTPUT_TOKENS, protocol["pricing"]) for a, c in jobs)
        return {"mode": "dry_run", "items": len(items), "requests": len(jobs), "api_requests_sent": 0,
                "worst_case_usd": round(bound, 3)}
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Protocol changed since the first session; start a new output")
    else:
        write_json(protocol_path, protocol)
        (output / "source").mkdir()
        (output / "source" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    ledger = RequestLedger(output, protocol["pricing"], budget_usd, max_requests, min_gap_seconds, caller)
    if ledger.stopped and not resume_after_stop:
        raise RuntimeError("Run was stopped earlier (see api_stop.json); inspect it, then pass --resume-after-stop to "
                           "continue with requests that were never started. Failed requests are not retried.")
    if resume_after_stop:
        ledger.resume_after_stop()
    records_path = output / "records.jsonl"
    done = {(r["asin"], r["condition"]) for r in read_jsonl(records_path)} if records_path.exists() else set()
    for asin, condition in jobs:
        if (asin, condition) in done:
            continue
        if ledger.stopped:
            break
        use_neighbors, use_image = CONDITIONS[condition]
        neighbors = finder.find(asin)
        image = image_entry(asin) if use_image else None
        parts, own, neighbor_texts = build_request(by_asin[asin], neighbors, use_neighbors, use_image, finder, image)
        response = ledger.call(f"{asin}_{condition}", parts, OUTPUT_TOKENS, json_response=True)
        if response["status"] == "not_attempted":
            break
        try:
            prediction = parse_prediction(response["text"], own, neighbor_texts) if response.get("text") else None
            status = "ok" if prediction is not None else response["status"]
        except (ValueError, TypeError):
            prediction, status = None, "invalid_json"
        append_jsonl(records_path, {"asin": asin, "condition": condition, "status": status, "prediction": prediction,
                                    "neighbors": neighbors if use_neighbors else [], "image_sent": bool(image),
                                    "usage": response.get("usage")})
    return {"requests_started": len(ledger.started_events()), "estimated_spend_usd": round(ledger.spent_usd(), 6),
            "stopped": ledger.stopped}


def analyze(output, products, split):
    records = read_jsonl(output / "records.jsonl")
    ok = [r for r in records if r["status"] == "ok"]
    selection = select_items(products)
    by_asin = {p["parent_asin"]: p for p in products}
    baseline = [{"asin": a, "condition": "rules", "prediction": rule_baseline(by_asin[a])} for a in selection[split]]
    cells = score_records(ok + baseline, products)
    excluded = {a: sum(1 for asin in selection[split] if a in gold_attributes(by_asin[asin]) and not valid_gold(a, gold_attributes(by_asin[asin])[a]))
                for a in ATTRIBUTES}
    result = {"split": split, "records": len(records), "usable": len(ok), "matching_rules": MATCHING_RULES,
              "invalid_gold_excluded": excluded,
              "invalid_or_failed": len(records) - len(ok), "summary": summarize(cells),
              "paired_differences": paired_differences(cells),
              "limitations": ["Gold is seller-entered and can itself be wrong or incomplete.",
                              "Lenient matching for color, brand and operating system; see matches().",
                              "Rule baseline is crude by design; it is a floor, not a competitive extractor.",
                              "Neighbors are retrieved by BM25 over listing text and may be unrelated products.",
                              "One model (gpt-4.1-mini); a small designed sample, not the catalog distribution."]}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--resume-after-stop", action="store_true")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    products = read_jsonl(SPLIT_DIRECTORY / "products.jsonl")
    output = args.output or OUTPUT_ROOT / f"results_spec_extraction_{args.split}_v1"
    if args.analyze:
        print(json.dumps(analyze(output, products, args.split)["summary"], indent=2))
        return
    if args.live and (not args.budget_usd or not args.max_requests):
        parser.error("--live needs explicit positive --budget-usd and --max-requests")
    caller = partial(openai_client.call_openai, model=MODEL)
    print(json.dumps(run(args.split, output, products, budget_usd=args.budget_usd, max_requests=args.max_requests,
                         live=args.live, caller=caller,
                         resume_after_stop=args.resume_after_stop), indent=2))


if __name__ == "__main__":
    main()
