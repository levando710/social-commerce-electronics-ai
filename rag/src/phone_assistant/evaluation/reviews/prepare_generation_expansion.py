"""Prepare paired generation cases and a cost scenario, without sending API requests."""
from phone_assistant.common.project_layout import source_path
from phone_assistant.common.project_layout import asset_path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from phone_assistant.llm.gemini_client import build_prompt
from phone_assistant.common.image_mime import image_mime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256

OUTPUT = ROOT / 'artifacts/evaluation/prepared_rag_generation_expansion_v1'
PROTOCOL = ROOT / 'data/discovery/RAG_GENERATION_EXPANSION_PLAN_2026_11_08.md'
SOURCES = {
    "development": ROOT / 'artifacts/evaluation/results_handset_rag_dev_v2/cases.jsonl',
    "previously_evaluated_test": ROOT / 'artifacts/evaluation/results_handset_rag_test_v2/cases.jsonl',
}
ARMS = ("text_only", "text_image")
IMAGE_LIMIT = 2
GENERATION_TOKEN_LIMIT = 1024
JUDGMENT_TOKEN_LIMIT = 3072
SCENARIO_INPUT_TOKENS_PER_REQUEST = 6000
TOKENS_PER_MILLION = 1_000_000
INPUT_USD_PER_MILLION = 0.30
OUTPUT_USD_PER_MILLION = 2.50
MODEL = "gemini-2.5-flash"
VISUAL_QUESTION = (
    "Trong các sản phẩm được cung cấp, hãy mô tả màu sắc và vị trí camera nhìn thấy ở một sản phẩm, "
    "nếu có bằng chứng rõ. Ghi [doc_id]. Nếu thiếu ảnh hoặc văn bản mô tả, nói chưa xác minh được. "
    "Chỉ mô tả hình dáng trực tiếp; không suy ra chất lượng camera hoặc tình trạng máy từ ảnh catalog."
)
CONDITION_QUESTION = (
    "Dựa đúng vào bằng chứng được cung cấp, có thể xác nhận một chiếc máy cụ thể còn 95% sức khỏe pin "
    "và chưa từng thay màn hình không? Nêu giới hạn bằng chứng, ghi [doc_id] nếu dẫn sản phẩm. "
    "Không coi ảnh catalog là ảnh kiểm định của chiếc máy đang bán."
)
EVIDENCE_INSTRUCTION = (
    "\nChỉ dùng nguồn được cung cấp. Có thể mô tả màu và vị trí camera trực tiếp nhìn thấy, "
    "nhưng không suy ra thông số ẩn, sức khỏe pin hay lịch sử sửa chữa. "
    "Nếu CONTEXT rỗng, giải thích chưa có sản phẩm/bằng chứng phù hợp; không tự tạo nguồn."
)


def canonical_rows(rows):
    """Use one predetermined Vietnamese formulation per intent, including empty contexts."""
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        query_id = row["query_id"]
        if query_id in ids:
            raise ValueError("Duplicate source query ID")
        ids.add(query_id)
        groups[row["intent_id"]].append(row)
    selected = []
    for intent_id, candidates in sorted(groups.items()):
        matches = [row for row in candidates
                   if row["query_id"].endswith(("_vi_0", "_vi"))]
        if len(matches) != 1:
            raise ValueError(f"Expected one canonical Vietnamese query for {intent_id}")
        row = matches[0]
        if row["kind"] not in ("retrieval", "expected_abstention"):
            raise ValueError("Unknown source task kind")
        if row["kind"] == "retrieval" and not row["contexts"]:
            raise ValueError("Retrieval source unexpectedly has no evidence")
        if row["kind"] == "expected_abstention" and row["contexts"]:
            raise ValueError("Expected abstention source unexpectedly contains candidates")
        selected.append(row)
    return selected


def image_manifest(contexts):
    images = []
    for context in contexts:
        for raw_path in context.get("product_images") or []:
            path = asset_path(raw_path)
            if not path.is_file():
                continue
            # Fail on corrupt evidence rather than silently changing the image arm.
            mime = image_mime(path)
            images.append({"image_id": f"image_{len(images) + 1}",
                           "doc_id": context["doc_id"], "path": str(path.resolve()),
                           "sha256": sha256(path), "mime_type": mime})
            break
        if len(images) >= IMAGE_LIMIT:
            break
    return images


def build_cases(source_rows):
    cases = []
    for cohort, rows in sorted(source_rows.items()):
        for row in canonical_rows(rows):
            questions = {"original": row["query"]}
            if row["kind"] == "retrieval":
                questions.update(visual=VISUAL_QUESTION, condition=CONDITION_QUESTION)
            images = image_manifest(row["contexts"])
            if row["kind"] == "retrieval" and not images:
                raise ValueError("Image comparison requires at least one valid supplied image")
            for question_type, question in questions.items():
                prompt = build_prompt(question, row["contexts"]) + EVIDENCE_INSTRUCTION
                pair_id = f"{cohort}_{row['query_id']}_{question_type}"
                for arm in ARMS:
                    cases.append({
                        "case_id": f"{pair_id}_{arm}", "pair_id": pair_id,
                        "source_id": row["query_id"], "intent_id": row["intent_id"],
                        "cohort": cohort, "question_type": question_type,
                        "source_kind": row["kind"], "arm": arm, "question": question,
                        "contexts": row["contexts"], "prompt": prompt,
                        "images": images if arm == "text_image" else [],
                    })
    validate_pairs(cases)
    return cases


def validate_pairs(cases):
    pairs = defaultdict(list)
    ids = set()
    for row in cases:
        if row["case_id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(row["case_id"])
        pairs[row["pair_id"]].append(row)
    for rows in pairs.values():
        if len(rows) != len(ARMS) or {r["arm"] for r in rows} != set(ARMS):
            raise ValueError("Every comparison must contain both arms exactly once")
        left, right = rows
        for field in ("question", "prompt", "contexts", "source_id", "cohort", "question_type"):
            if left[field] != right[field]:
                raise ValueError(f"Paired evidence changed: {field}")
        if any(row["images"] for row in rows if row["arm"] == "text_only"):
            raise ValueError("Text-only arm must not receive images")


def cost_scenario(answer_count):
    maximum_requests = answer_count * len(("generation", "judgment"))
    assumed_input = maximum_requests * SCENARIO_INPUT_TOKENS_PER_REQUEST
    capped_output = answer_count * (GENERATION_TOKEN_LIMIT + JUDGMENT_TOKEN_LIMIT)
    estimated_usd = (assumed_input * INPUT_USD_PER_MILLION
                     + capped_output * OUTPUT_USD_PER_MILLION) / TOKENS_PER_MILLION
    return {"model": MODEL, "planned_answers": answer_count,
            "maximum_requests_no_retries": maximum_requests,
            "assumed_input_tokens_per_request_including_images": SCENARIO_INPUT_TOKENS_PER_REQUEST,
            "scenario_input_tokens": assumed_input, "capped_output_tokens": capped_output,
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "scenario_usd": round(estimated_usd, 6),
            "scope": "Planning scenario, not a price guarantee or enforced spend cap. Input tokens are assumed, not measured; no API countTokens call. Excludes taxes, retries, extra experiments and other services.",
            "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
            "pricing_checked_date": "2026-11-08"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(output):
    if output.exists():
        raise FileExistsError("Preserve the prepared protocol; use a new output to revise it")
    cases = build_cases({name: read_jsonl(path) for name, path in SOURCES.items()})
    budget = cost_scenario(len(cases))
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")
    write_json(output / "budget.json", budget)
    sources = (Path(__file__), source_path('gemini_client.py'),
               source_path('image_mime.py'), source_path('recsys_data.py'))
    snapshot = output / "source"
    snapshot.mkdir()
    for path in sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    manifest = {
        "status": "prepared_not_executed", "api_requests_sent": 0,
        "planned_answers": len(cases), "paired_questions": len(cases) // len(ARMS),
        "source_groups": len({(row["cohort"], row["intent_id"]) for row in cases}),
        "cases_sha256": sha256(cases_path), "budget_sha256": sha256(output / "budget.json"),
        "protocol_sha256": sha256(PROTOCOL),
        "source_data_sha256": {name: sha256(path) for name, path in SOURCES.items()},
        "source_code_sha256": {path.name: sha256(path) for path in sources},
        "answer_counts_by_type": dict(Counter(row["question_type"] for row in cases)),
        "limitations": ["Researcher-authored prompts and catalog contexts, not independent human conversations.",
                        "Existing development and previously inspected test cohorts; not a new untouched confirmation set.",
                        "Count matched source groups, not paraphrases/arms as independent users.",
                        "Empty-evidence image arms are negative controls, not visual ablation evidence.",
                        "No answers, automatic quality grades or human labels have been produced."]}
    write_json(output / "manifest.json", manifest)
    write_json(output / "preflight.json", {
        "paired_question_context_parity": "passed", "unique_case_ids": "passed",
        "images_verified": len({image["sha256"] for row in cases for image in row["images"]}),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "api_requests_sent": 0})
    return manifest, budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    manifest, budget = prepare(parser.parse_args().output)
    print(json.dumps({"status": manifest["status"], "source_groups": manifest["source_groups"],
                      **budget}, indent=2))


if __name__ == "__main__":
    main()
