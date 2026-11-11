import argparse
import csv
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests


from phone_pricing.common.project_paths import RAW_PHONE_DATA, SCRAPE_LOG, configure_utf8_output, ensure_project_dirs

configure_utf8_output()


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
]

PHONE_FIELDS = [
    "ad_id",
    "subject",
    "price",
    "brand",
    "model",
    "capacity",
    "color",
    "condition",
    "warranty",
    "origin",
    "lock_status",
    "body",
]

REGIONS = [
    12000,  # Hà Nội
    13000,  # TP.HCM
    7000,
    11000,
    14000,
    15000,
    2000,
    3000,
    4000,
]

POPULAR_BRAND_TARGETS = {
    "Apple": 3500,
    "Samsung": 2500,
    "Xiaomi": 2000,
    "Oppo": 1200,
    "Vivo": 1000,
    "Honor": 800,
    "Huawei": 700,
    "Google": 600,
    "Realme": 600,
    "Sony": 400,
}

GENERAL_TARGET = 10000
OUTPUT_FILE = RAW_PHONE_DATA
SCRAPE_LOG_FILE = SCRAPE_LOG


def normalize_brand(value):
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split()).lower()


def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
        "Origin": "https://www.chotot.com",
        "Referer": "https://www.chotot.com/",
    }


def extract_phone_text(list_id, ad, params_dict):
    return {
        "ad_id": list_id,
        "subject": " ".join(str(ad.get("subject", "")).split()),
        "price": ad.get("price", 0),
        "brand": params_dict.get("mobile_brand", ""),
        "model": params_dict.get("mobile_model", ""),
        "capacity": params_dict.get("mobile_capacity", ""),
        "color": params_dict.get("mobile_color") or params_dict.get("color", ""),
        "condition": params_dict.get("elt_condition", ""),
        "warranty": params_dict.get("elt_warranty", ""),
        "origin": params_dict.get("elt_origin", ""),
        "lock_status": params_dict.get("elt_lock", ""),
        "body": " ".join(str(ad.get("body", "")).split()),
    }


def request_json(url, *, params=None, retry=3):
    for attempt in range(retry):
        try:
            res = requests.get(url, headers=get_random_headers(), params=params, timeout=12)
            if res.status_code == 429:
                sleep_seconds = 20 + attempt * 15
                print(f"Bị rate-limit, nghỉ {sleep_seconds}s rồi thử lại...")
                time.sleep(sleep_seconds)
                continue
            if res.status_code != 200:
                return None
            return res.json()
        except requests.RequestException:
            time.sleep(3 + attempt * 2)
    return None


def fetch_detail(list_id):
    url = f"https://gateway.chotot.com/v1/public/ad-listing/{list_id}"
    data = request_json(url)
    if not data:
        return None

    ad = data.get("ad", {})
    if not ad:
        return None

    params_list = data.get("parameters", []) or ad.get("parameters", [])
    params_dict = {
        p.get("id"): str(p.get("value")).strip()
        for p in params_list
        if p.get("id") and p.get("value") is not None
    }
    return extract_phone_text(list_id, ad, params_dict)


def load_existing(output_file):
    seen_ids = set()
    brand_counts = Counter()
    row_count = 0

    if not os.path.exists(output_file):
        return seen_ids, brand_counts, row_count

    with open(output_file, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen_ids.add(int(row["ad_id"]))
            except (TypeError, ValueError):
                continue
            row_count += 1
            brand_counts[normalize_brand(row.get("brand", ""))] += 1

    return seen_ids, brand_counts, row_count


def append_record(output_file, record):
    ensure_project_dirs()
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    with open(output_file, mode="a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PHONE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in PHONE_FIELDS})


def append_log(log_file, record, query, region_id, brand_target):
    ensure_project_dirs()
    file_exists = os.path.exists(log_file) and os.path.getsize(log_file) > 0
    fields = ["scraped_at", "ad_id", "brand", "query", "region_id", "brand_target"]
    with open(log_file, mode="a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "ad_id": record.get("ad_id", ""),
                "brand": record.get("brand", ""),
                "query": query,
                "region_id": region_id,
                "brand_target": brand_target or "",
            }
        )


def backup_existing_file(path):
    if not os.path.exists(path):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{timestamp}"
    os.replace(path, backup_path)
    return backup_path


def build_brand_queries(brand):
    base = [brand, f"{brand} cũ", f"{brand} 128GB", f"{brand} 256GB"]
    damage = [f"{brand} xác", f"{brand} hỏng", f"{brand} icloud"]
    return base + damage


def scrape_query(output_file, seen_ids, brand_counts, query, *, brand_target=None, need=100, max_pages=30):
    list_url = "https://gateway.chotot.com/v1/public/ad-listing"
    accepted = 0
    target_norm = normalize_brand(brand_target)

    for region_id in REGIONS:
        if accepted >= need:
            break

        for page in range(max_pages):
            if accepted >= need:
                break

            params = {
                "cg": 5010,
                "limit": 50,
                "o": page * 50,
                "region_v2": region_id,
                "st": "s,k",
            }
            if query:
                params["q"] = query

            data = request_json(list_url, params=params)
            if not data:
                break

            ads = data.get("ads", [])
            if not ads:
                break

            new_ids_on_page = 0
            for item in ads:
                list_id = item.get("list_id") or item.get("ad_id")
                if not list_id or list_id in seen_ids:
                    continue

                seen_ids.add(list_id)
                new_ids_on_page += 1
                record = fetch_detail(list_id)
                time.sleep(random.uniform(0.7, 1.8))

                if not record or not record.get("price") or int(record.get("price", 0)) <= 0:
                    continue

                record_brand_norm = normalize_brand(record.get("brand", ""))
                if target_norm and record_brand_norm != target_norm:
                    continue

                append_record(output_file, record)
                append_log(SCRAPE_LOG_FILE, record, query, region_id, brand_target)
                brand_counts[record_brand_norm] += 1
                accepted += 1

                if accepted % 25 == 0:
                    label = brand_target or "general"
                    print(f"[{label}] +{accepted}/{need} | query='{query}' | region={region_id} | page={page}")

            if new_ids_on_page == 0:
                break

            time.sleep(random.uniform(1.0, 2.5))

    return accepted


def audit_dataset(output_file=OUTPUT_FILE):
    _, brand_counts, row_count = load_existing(output_file)

    print(f"\nTổng số bài hiện có: {row_count}")
    print("\nPhân bố hãng phổ biến:")
    for brand, target in POPULAR_BRAND_TARGETS.items():
        count = brand_counts[normalize_brand(brand)]
        status = "OK" if count >= target else f"thiếu {target - count}"
        print(f"- {brand:8s}: {count:5d}/{target:<5d} {status}")

    missing = {
        brand: target - brand_counts[normalize_brand(brand)]
        for brand, target in POPULAR_BRAND_TARGETS.items()
        if brand_counts[normalize_brand(brand)] < target
    }
    return missing


def run_scraper(output_file=OUTPUT_FILE, dry_run=False, fresh=False):
    ensure_project_dirs()
    if fresh and not dry_run:
        data_backup = backup_existing_file(output_file)
        log_backup = backup_existing_file(SCRAPE_LOG_FILE)
        if data_backup:
            print(f"Đã backup dữ liệu cũ: {data_backup}")
        if log_backup:
            print(f"Đã backup log cũ: {log_backup}")

    seen_ids, brand_counts, row_count = load_existing(output_file)
    print(f"Đã có {row_count} bài trong {output_file}.")

    if row_count < GENERAL_TARGET:
        need = GENERAL_TARGET - row_count
        print(f"\nCào tổng quan thêm {need} bài trước khi cào bù theo hãng...")
        if not dry_run:
            general_queries = ["", "điện thoại cũ", "xác", "hỏng màn", "icloud"]
            for query in general_queries:
                if row_count >= GENERAL_TARGET:
                    break
                accepted = scrape_query(output_file, seen_ids, brand_counts, query, need=GENERAL_TARGET - row_count)
                row_count += accepted

    missing = audit_dataset(output_file)
    if not missing:
        print("\nCác hãng phổ biến đã đạt quota.")
        return

    print("\nKế hoạch cào bù:")
    for brand, need in missing.items():
        print(f"- {brand}: cần thêm {need} bài")

    if dry_run:
        return

    for brand, need in missing.items():
        remaining = need
        print(f"\nBắt đầu cào bù hãng {brand}...")
        for query in build_brand_queries(brand):
            if remaining <= 0:
                break
            accepted = scrape_query(
                output_file,
                seen_ids,
                brand_counts,
                query,
                brand_target=brand,
                need=remaining,
                max_pages=40,
            )
            remaining -= accepted
            print(f"{brand}: query='{query}' lấy thêm {accepted}, còn thiếu {max(remaining, 0)}")

        if remaining > 0:
            print(f"Cảnh báo: {brand} vẫn thiếu {remaining} bài sau khi quét các query hiện có.")

    audit_dataset(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cào dữ liệu điện thoại và tự cào bù các hãng phổ biến còn thiếu.")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="File CSV dữ liệu thô.")
    parser.add_argument("--audit", action="store_true", help="Chỉ kiểm tra phân bố hãng, không cào.")
    parser.add_argument("--dry-run", action="store_true", help="In kế hoạch cào bù, không gọi API.")
    parser.add_argument("--fresh", action="store_true", help="Backup file cũ rồi cào lại từ đầu.")
    args = parser.parse_args()

    if args.audit:
        audit_dataset(args.output)
    else:
        run_scraper(args.output, dry_run=args.dry_run, fresh=args.fresh)
