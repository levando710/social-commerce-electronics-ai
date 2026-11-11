"""Collect a bounded phone corpus from the public Amazon Reviews 2023 release."""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlparse


ROOT = MODULE_ROOT
BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
SOURCES = {
    "metadata": BASE + "/meta_categories/meta_Cell_Phones_and_Accessories.jsonl",
    "reviews": BASE + "/review_categories/Cell_Phones_and_Accessories.jsonl",
}
ACCESSORY_LEAVES = {
    "accessories", "cases, holsters & sleeves", "iphone accessories", "basic cases",
    "sleeves", "armbands", "screen protectors", "replacement parts", "batteries",
    "chargers & power adapters", "charging stations", "cables & adapters",
    "wireless chargers", "wall chargers", "portable power banks", "stylus pens",
    "stands", "mounts", "cradles", "holsters", "wallets", "phone charms",
    "anti-dust plugs", "sim card tools & accessories", "lens attachments", "tripods",
    "handheld gimbals & stabilizers", "selfie sticks", "flashes & selfie lights",
    "selfie lights", "screen expanders & magnifiers", "signal boosters",
    "virtual reality (vr) headsets", "single ear bluetooth headsets",
    "smartwatch cables & chargers", "uv phone sterilizer boxes", "grips",
    "triggers", "pads & mats", "maintenance, upkeep & repairs",
}
PHONE_LEAVES = {
    "cell phones", "smartphones", "unlocked cell phones", "unlocked wireless trade-in",
    "wireless trade-in", "carrier branded wireless trade-in", "nokia", "verizon wireless",
}

ACCESSORY_TITLE = re.compile(
    r"\b("
    r"case|cases|cover|covers|coveron|flip\s*cover|screen\s*protectors?|tempered\s*glass|protectors?|"
    r"charger|charging\s*(?:dock|station|stand|pad)|cable|adapter|otg|"
    r"holder|mount|stand|tripod|grip|lanyard|strap|armband|holster|pouch|wallet|"
    r"skin|sleeve|shield|housing|shell|faceplate|door|card\s*reader|memory\s*card\s*reader|"
    r"stylus|pen|earbuds|earphones|headphones|headset|speaker|"
    r"replacement|spare|repair\s*(?:part|kit)|lcd|digitizer|screen\s*replacement|"
    r"battery\s*(?:replacement|for|case|pack|door)|power\s*bank|"
    r"lens|selfie\s*(?:stick|light)|gimbal|stabilizer|"
    r"sim\s*(?:card|tray|tool|kit)|prepaid|minutes|"
    r"signal\s*booster|repeater|amplifier|antenna|"
    r"vr\s*headset|virtual\s*reality|cardboard|"
    r"smartwatch|watch|tablet|dummy|replica|toy|decor|cleaner|sterilizer"
    r")\b",
    re.I,
)
ACCESSORY_PHRASE = re.compile(
    r"\b(?:for|compatible with|fits?)\s+(?:apple\s+)?iphone\b|"
    r"\b(?:for|compatible with|fits?)\s+samsung\s+galaxy\b|"
    r"\b(?:for|compatible with|fits?)\s+(?:google\s+)?pixel\b",
    re.I,
)
BUNDLED_ACCESSORY_PHRASE = re.compile(
    r"\b(?:with|includes?|included|bundle(?:d)?\s+with)\s+(?:a\s+)?"
    r"(?:case|cover|charger|charging\s+cable|cable|adapter|screen\s*protector)\b",
    re.I,
)
ACCESSORY_TOOL_PHRASE = re.compile(r"\b(?:cover\s*)?(?:removal|pry)\s+tool\b", re.I)
PHONE_OBJECT = re.compile(
    r"\b(smart\s?phone|cell\s?phone|mobile\s*phone|android\s*phone|windows\s*phone|"
    r"feature\s*phone|flip\s*phone|bar\s*phone|gsm\s*phone|cdma\s*phone|"
    r"unlocked\s*(?:phone|cell|smartphone)|4g\s*lte\s*(?:phone|smartphone))\b",
    re.I,
)
PHONE_MODEL = re.compile(
    r"\b("
    r"(?:apple\s+)?iphone\s*(?:se|xs|max|xr|x|[3-9]|1[0-9])(?:\s*(?:mini|pro|max|plus|promax|pro\s*max))?|"
    r"samsung\s+galaxy\s*(?:s|a|m|f|j|z|note|fold|flip)\s*[\w+]*|"
    r"(?:google\s+)?pixel\s*(?:fold|[1-9]\w*)|"
    r"one\s*plus\s*\w+|oneplus\s*\w+|"
    r"moto(?:rola)?\s*(?:g|e|z|x|edge|razr)\s*[\w+]*|"
    r"nokia\s*[\w-]+|lg\s*(?:g|v|k|stylo|optimus|nexus|ku|vx)\s*[\w-]+|"
    r"sony\s*ericsson\s*[\w-]+|sony\s*xperia\s*[\w-]+|"
    r"blackberry\s*[\w-]+|palm\s*(?:treo|pre|pixi)\s*[\w-]+|"
    r"htc\s*(?:one|desire|evo|wildfire|inspire|aria|sensation|touch)\s*[\w-]*|"
    r"zte\s*[\w-]+|alcatel\s*[\w-]+|tcl\s*[\w-]+|blu\s*[\w-]+|"
    r"huawei\s*[\w-]+|honor\s*[\w-]+|xiaomi\s*[\w-]+|redmi\s*(?:note\s*)?[\w-]+|"
    r"oppo\s*[\w-]+|vivo\s*[\w-]+|realme\s*[\w-]+|asus\s*rog\s*phone\s*[\w-]*"
    r")\b",
    re.I,
)
MODEM_OR_NETWORK = re.compile(r"\b(usb\s*)?modem|hotspot|router|mifi|aircard\b", re.I)


def phone_decision(row):
    title = str(row.get("title") or "").strip()
    categories = row.get("categories") or []
    if not title:
        return False, "missing_title"
    category_path = [str(c).strip().lower() for c in categories if str(c).strip()]
    leaf = category_path[-1] if category_path else ""
    phone_model_match = PHONE_MODEL.search(title)
    has_phone_model = bool(phone_model_match)
    has_phone_object = bool(PHONE_OBJECT.search(title))

    if MODEM_OR_NETWORK.search(title):
        return False, "network_device_title"
    if any(c in ACCESSORY_LEAVES for c in category_path[1:]):
        return False, "accessory_category"
    accessory_match = ACCESSORY_TITLE.search(title)
    if accessory_match:
        # A real phone listing may say "with case" as an included item. Accessory listings
        # put accessory words before the phone model, such as "case for iPhone".
        bundled_match = BUNDLED_ACCESSORY_PHRASE.search(title)
        bundled_after_model = (
            phone_model_match and bundled_match and
            bundled_match.start() > phone_model_match.start() and
            accessory_match.start() >= bundled_match.start()
        )
        if not (bundled_after_model and not ACCESSORY_PHRASE.search(title) and
                not ACCESSORY_TOOL_PHRASE.search(title)):
            return False, "accessory_or_ambiguous_title"

    if has_phone_model:
        return True, "phone_model_title"
    if has_phone_object and (leaf in PHONE_LEAVES or "cell phones" in category_path):
        return True, "phone_object_title"
    return False, "needs_manual_review"


def image_urls(value):
    """Accept original JSONL lists and column-oriented image dictionaries."""
    if isinstance(value, dict):
        if any(isinstance(v, list) for v in value.values()):
            count = max((len(v) for v in value.values() if isinstance(v, list)), default=0)
            value = [{k: v[i] if isinstance(v, list) and i < len(v) else None
                      for k, v in value.items()} for i in range(count)]
        else:
            value = [value]
    result = []
    for entry in value or []:
        if not isinstance(entry, dict):
            continue
        url = next((entry.get(k) for k in ("hi_res", "large", "large_image_url",
                                          "medium_image_url", "thumb") if entry.get(k)), None)
        if isinstance(url, str) and url.startswith("https://") and url not in result:
            result.append(url)
    return result


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS products (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS review_parent ON reviews(parent_id);
        CREATE TABLE IF NOT EXISTS state (stage TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit (
            id TEXT PRIMARY KEY, title TEXT, reason TEXT, accepted INTEGER);
        CREATE TABLE IF NOT EXISTS images (
            url TEXT PRIMARY KEY, path TEXT, error TEXT);
    """)
    return db


@contextmanager
def source_stream(source, offset, etag):
    if not source.startswith("https://"):
        with Path(source).open("rb") as stream:
            stat = Path(source).stat()
            identity = f"local:{stat.st_size}:{stat.st_mtime_ns}"
            if etag and identity != etag:
                raise ValueError("Local source changed; use a new --output directory.")
            stream.seek(offset)
            yield stream, identity
        return
    import requests
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if etag:
            headers["If-Range"] = etag
    with requests.get(source, headers=headers, stream=True, timeout=(20, 90)) as response:
        if (offset and response.status_code == 416 and
                response.headers.get("Content-Range") == f"bytes */{offset}" and
                etag and response.headers.get("ETag") == etag):
            yield io.BytesIO(), etag
            return
        response.raise_for_status()
        if offset and (response.status_code != 206 or not
                       response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ValueError("Server cannot resume this source safely. Use a local JSONL or new output.")
        identity = response.headers.get("ETag")
        if etag and identity != etag:
            raise ValueError("Remote source changed; use a new --output directory.")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Compressed HTTP response cannot use byte checkpoints.")
        yield io.BufferedReader(response.raw), identity


def collect(db, stage, args):
    source = getattr(args, stage + "_source") or SOURCES[stage]
    saved = db.execute("SELECT payload FROM state WHERE stage=?", (stage,)).fetchone()
    state = json.loads(saved[0]) if saved else {
        "source": source, "offset": 0, "lines": 0, "etag": None, "eof": False,
        "counts": {},
    }
    if state["source"] != source:
        raise ValueError("Source differs from checkpoint; use a new --output directory.")
    if state["eof"]:
        print(f"{stage}: source already scanned completely.")
        return
    ids = {r[0] for r in db.execute("SELECT id FROM products")}
    if stage == "reviews" and not ids:
        raise ValueError("Collect metadata before collecting reviews.")
    counts = Counter(dict(db.execute("SELECT parent_id, COUNT(*) FROM reviews GROUP BY parent_id")))
    remaining = {parent for parent in ids if counts[parent] < args.reviews_per_product}
    if stage == "reviews" and not remaining:
        print("Review targets already reached.")
        return
    if stage == "metadata" and len(ids) >= args.max_products:
        print("Product target already reached.")
        return
    processed = 0
    reasons = Counter(state["counts"])
    def checkpoint():
        state["counts"] = dict(reasons)
        db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (stage, json.dumps(state)))
        db.commit()
    try:
        with source_stream(source, state["offset"], state["etag"]) as (stream, identity):
            state["etag"] = identity
            while processed < args.scan_limit:
                line = stream.readline()
                if not line:
                    state["eof"] = True
                    break
                # Advance only after parsing and processing the complete record succeeds.
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at byte {state['offset']}")
                parent = str(row.get("parent_asin") or "")
                if stage == "metadata":
                    keep, reason = phone_decision(row)
                    if not parent:
                        keep, reason = False, "missing_parent_asin"
                    reasons[reason] += 1
                    audit_id = parent or f"line:{state['lines']}"
                    db.execute("INSERT OR REPLACE INTO audit VALUES (?, ?, ?, ?)",
                               (audit_id, str(row.get("title") or ""), reason, int(keep)))
                    if keep:
                        db.execute("INSERT OR IGNORE INTO products VALUES (?, ?)",
                                   (parent, json.dumps(row, ensure_ascii=False)))
                        ids.add(parent)
                elif parent in ids and counts[parent] < args.reviews_per_product:
                    if str(row.get("text") or "").strip():
                        key = json.dumps([parent, row.get("asin"), row.get("user_id"),
                                          row.get("timestamp"), row.get("title"), row.get("text")])
                        rid = hashlib.sha256(key.encode()).hexdigest()
                        cur = db.execute("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?)",
                                         (rid, parent, json.dumps(row, ensure_ascii=False)))
                        counts[parent] += cur.rowcount
                        if counts[parent] >= args.reviews_per_product:
                            remaining.discard(parent)
                state["offset"] += len(line)
                state["lines"] += 1
                processed += 1
                if processed % 1000 == 0:
                    checkpoint()
                    print(f"{stage}: scanned {state['lines']:,}; products {len(ids):,}; "
                          f"reviews {sum(counts.values()):,}", flush=True)
                if stage == "metadata" and len(ids) >= args.max_products:
                    break
                if stage == "reviews" and not remaining:
                    break
    finally:
        checkpoint()
    print(f"{stage}: stopped at line {state['lines']:,}; EOF={state['eof']}")


def download_images(db, args):
    import requests
    from PIL import Image
    folder = args.output / "images"
    folder.mkdir(parents=True, exist_ok=True)
    attempted = 0
    with requests.Session() as session:
        for table in ("products", "reviews"):
            for (payload,) in db.execute(f"SELECT payload FROM {table}"):
                row = json.loads(payload)
                for url in image_urls(row.get("images"))[:args.images_per_record]:
                    saved = db.execute("SELECT path, error FROM images WHERE url=?", (url,)).fetchone()
                    if saved and not saved[1] and (args.output / saved[0]).exists():
                        continue
                    if attempted >= args.max_images:
                        return
                    attempted += 1
                    target = folder / (hashlib.sha256(url.encode()).hexdigest() + ".jpg")
                    try:
                        host = (urlparse(url).hostname or "").lower()
                        if not (host.endswith(".media-amazon.com") or host.endswith(".ssl-images-amazon.com")):
                            raise ValueError("Unsupported image host")
                        with session.get(url, stream=True, timeout=(15, 30)) as response:
                            response.raise_for_status()
                            data = bytearray()
                            for chunk in response.iter_content(65536):
                                data.extend(chunk)
                                if len(data) > 15 * 1024 * 1024:
                                    raise ValueError("Image exceeds 15 MB")
                        with Image.open(io.BytesIO(data)) as picture:
                            picture.verify()
                        with Image.open(io.BytesIO(data)) as picture:
                            picture.convert("RGB").save(target.with_suffix(".part"), format="JPEG", quality=95)
                        target.with_suffix(".part").replace(target)
                        db.execute("INSERT OR REPLACE INTO images VALUES (?, ?, NULL)",
                                   (url, target.relative_to(args.output).as_posix()))
                    except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError) as exc:
                        db.execute("INSERT OR REPLACE INTO images VALUES (?, NULL, ?)", (url, str(exc)))
                    db.commit()
                    if attempted % 25 == 0:
                        print(f"Images attempted: {attempted}", flush=True)
                    time.sleep(0.2)


def export(db, args):
    folder = args.output / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    for table in ("products", "reviews"):
        target = folder / f"{table}.jsonl"
        with target.with_suffix(".part").open("w", encoding="utf-8") as out:
            for rid, payload in db.execute(f"SELECT id, payload FROM {table} ORDER BY id"):
                row = json.loads(payload)
                row["record_id"] = rid
                row["source_dataset"] = "McAuley-Lab/Amazon-Reviews-2023"
                row["image_assets"] = []
                for url in image_urls(row.get("images")):
                    asset = db.execute("SELECT path, error FROM images WHERE url=?", (url,)).fetchone()
                    row["image_assets"].append({"url": url, "path": asset[0] if asset else None,
                                                "error": asset[1] if asset else "not_downloaded"})
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        target.with_suffix(".part").replace(target)
    import csv
    with (folder / "filter_audit.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["parent_asin", "title", "reason", "accepted"])
        writer.writerows(db.execute("SELECT * FROM audit ORDER BY id"))
    summary = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
               for table in ("products", "reviews")}
    summary["products_with_reviews"] = db.execute("SELECT COUNT(DISTINCT parent_id) FROM reviews").fetchone()[0]
    summary["images_ok"] = db.execute("SELECT COUNT(*) FROM images WHERE error IS NULL").fetchone()[0]
    summary["images_failed"] = db.execute("SELECT COUNT(*) FROM images WHERE error IS NOT NULL").fetchone()[0]
    summary["checkpoints"] = {s: json.loads(p) for s, p in db.execute("SELECT * FROM state")}
    summary["sampling"] = "First matching products and first nonempty unique reviews; not a random benchmark."
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoints"}, indent=2))


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["metadata", "reviews", "images", "export", "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--max-products", type=positive, default=1000)
    parser.add_argument("--reviews-per-product", type=positive, default=20)
    parser.add_argument("--scan-limit", type=positive, default=1_000_000,
                        help="Maximum source lines per stage per run; rerun to continue")
    parser.add_argument("--images-per-record", type=positive, default=2)
    parser.add_argument("--max-images", type=positive, default=2000,
                        help="Maximum image attempts per run")
    parser.add_argument("--metadata-source", help="Optional local uncompressed JSONL or HTTPS URL")
    parser.add_argument("--reviews-source", help="Optional local uncompressed JSONL or HTTPS URL")
    args = parser.parse_args()
    args.output = args.output.resolve()
    db = connect(args.output / "raw" / "amazon.sqlite3")
    try:
        if args.stage in ("metadata", "all"):
            collect(db, "metadata", args)
        if args.stage in ("reviews", "all"):
            collect(db, "reviews", args)
        if args.stage in ("images", "all"):
            download_images(db, args)
        export(db, args)
    except KeyboardInterrupt:
        print("Stopped. Committed data retained; rerun the same command to continue.")
        export(db, args)
    finally:
        db.close()


if __name__ == "__main__":
    main()
