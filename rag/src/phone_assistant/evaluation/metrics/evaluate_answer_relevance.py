"""Answer Relevance of the 104 generated answers (reverse-question method, as in Ragas).

For each saved answer a language model writes three short questions that the answer responds to (no access to the
original question or the evidence).  Relevance is the mean cosine between the embedding of the original question and
the embeddings of those reverse questions, with the same frozen multilingual MiniLM used elsewhere.  Answers the model
flags as non-committal (a refusal or "no information") are counted and reported separately, because for questions
that lack evidence a refusal is the intended behaviour; they are not silently set to zero.

Protocol: dry run by default; ``--live`` needs ``--budget-usd`` and ``--max-requests``; each request is journaled
before it is sent and never retried; the run stops at the first API error; ``--resume`` skips saved responses.
Scope: custom method with a same-provider language model, not the official Ragas implementation and not a human rating.
"""
from phone_assistant.common.project_layout import MODULE_ROOT
from phone_assistant.llm import openai_client
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

ROOT = MODULE_ROOT
EVALUATION = ROOT / "artifacts/evaluation"
SOURCE = EVALUATION / "results_rag_generation_expansion_openai_v1"
OUTPUT = EVALUATION / "results_answer_relevance_openai_v1"
MODEL = "gpt-4.1-mini"
TEXT_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"  # frozen MiniLM revision used in all experiments
PRICES = (0.40, 1.60)  # USD per million input / output tokens (same as the generation protocol)
MAX_OUTPUT_TOKENS = 400
WORST_CASE_INPUT_TOKENS = 1500
QUESTION_COUNT = 3
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261111
MIN_GAP_SECONDS = 1.0
PROMPT = (
    "Dưới đây là một câu trả lời. Hãy viết {count} câu hỏi ngắn, khác nhau, mà câu trả lời này đang trả lời. "
    "Không có thông tin nào khác ngoài câu trả lời. Nếu câu trả lời chỉ nói rằng không có thông tin hoặc không thể "
    "xác nhận, đặt noncommittal là true, ngược lại là false. Chỉ trả về JSON: "
    "{{\"questions\": [\"...\"], \"noncommittal\": false}}.\n\nCÂU TRẢ LỜI:\n{answer}")


def load_cases(source=SOURCE):
    cases = []
    for path in sorted(source.glob("case_*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("answer"):
            cases.append({key: case[key] for key in ("case_id", "pair_id", "arm", "question_type", "question", "answer")})
    return cases


def cost(usage):
    return (usage.get("promptTokenCount", 0) * PRICES[0] + (usage.get("candidatesTokenCount", 0)
            + usage.get("thoughtsTokenCount", 0)) * PRICES[1]) / 1e6


def worst_case_cost():
    return (WORST_CASE_INPUT_TOKENS * PRICES[0] + MAX_OUTPUT_TOKENS * PRICES[1]) / 1e6


def parse_questions(text):
    data = json.loads(text)
    questions = [str(item).strip() for item in data.get("questions", []) if str(item).strip()]
    if not questions:
        raise ValueError("no questions returned")
    return questions[:QUESTION_COUNT], bool(data.get("noncommittal"))


def read_journal(path):
    entries = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            entries[record["case_id"]] = record
    return entries


def run(cases, output, live, budget_usd, max_requests, call=openai_client.call_openai, sleep=time.sleep):
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "requests.jsonl"
    done = read_journal(journal)
    pending = [case for case in cases if case["case_id"] not in done]
    plan = {"cases": len(cases), "pending": len(pending), "worst_case_usd_per_request": worst_case_cost(),
            "worst_case_usd_total": round(worst_case_cost() * len(pending), 4), "model": MODEL, "live": live}
    if not live:
        return {"dry_run": True, **plan}
    if budget_usd is None or max_requests is None:
        raise SystemExit("--live needs --budget-usd and --max-requests")
    spent = sum(cost(record.get("usage") or {}) if record.get("status") == "returned" else worst_case_cost()
                for record in done.values())
    sent, stopped = 0, None
    for case in pending:
        if len(done) >= max_requests or spent + worst_case_cost() > budget_usd:
            stopped = "request or budget cap reached"
            break
        with journal.open("a", encoding="utf-8") as handle:  # journaled before the network call
            handle.write(json.dumps({"case_id": case["case_id"], "status": "started", "at": time.time()}) + "\n")
        response = call([{"text": PROMPT.format(count=QUESTION_COUNT, answer=case["answer"])}], temperature=0,
                        max_output_tokens=MAX_OUTPUT_TOKENS, timeout=60, response_mime_type="application/json", model=MODEL)
        record = {"case_id": case["case_id"], "at": time.time()}
        try:
            if response.get("error") or not response.get("text"):
                raise RuntimeError(response.get("error") or "empty response")
            questions, noncommittal = parse_questions(response["text"])
            record.update(status="returned", questions=questions, noncommittal=noncommittal, usage=response.get("usage"),
                          model=response.get("model_version") or response.get("model"))
            spent += cost(response.get("usage") or {})
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            record.update(status="failed", error=str(error)[:300])
            stopped = f"stopped at the first error: {record['error']}"
        done[case["case_id"]] = record
        (output / f"request_{case['case_id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        sent += 1
        if stopped:
            break
        sleep(MIN_GAP_SECONDS)
    return {**plan, "sent": sent, "spent_usd": round(spent, 4), "stopped": stopped}


def read_results(output):
    """Last journal record per case; 'started' without a result is an interrupted request and is not retried."""
    return read_journal(output / "requests.jsonl")


def bootstrap(deltas, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS):
    generator = random.Random(seed)
    means = sorted(sum(generator.choice(deltas) for _ in deltas) / len(deltas) for _ in range(repetitions))
    return means[int(0.025 * repetitions)], means[int(0.975 * repetitions) - 1]


def analyze(cases, output):
    from phone_assistant.retrieval.search.dense_search import TEXT_MODEL, load_encoder
    records = read_results(output)
    encoder = load_encoder(TEXT_MODEL, revision=TEXT_REVISION)
    rows = []
    for case in cases:
        record = records.get(case["case_id"])
        if not record or record.get("status") != "returned":
            continue
        vectors = encoder.encode([case["question"]] + record["questions"], normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        rows.append({**{key: case[key] for key in ("case_id", "pair_id", "arm", "question_type")},
                     "relevance": float((vectors[1:] @ vectors[0]).mean()), "noncommittal": record["noncommittal"],
                     "questions": record["questions"]})
    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None
    summary = {"answers_scored": len(rows), "answers_expected": len(cases), "by_arm": {}, "by_question_type": {}}
    for arm in sorted({row["arm"] for row in rows}):
        part = [row for row in rows if row["arm"] == arm]
        committed = [row["relevance"] for row in part if not row["noncommittal"]]
        summary["by_arm"][arm] = {"answers": len(part), "relevance_mean": mean(row["relevance"] for row in part),
                                  "noncommittal_answers": sum(row["noncommittal"] for row in part),
                                  "relevance_mean_committal_only": mean(committed)}
    for kind in sorted({row["question_type"] for row in rows}):
        summary["by_question_type"][kind] = {arm: mean(row["relevance"] for row in rows
                                             if row["question_type"] == kind and row["arm"] == arm)
                                             for arm in summary["by_arm"]}
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["arm"]] = row["relevance"]
    deltas = [pair["text_image"] - pair["text_only"] for pair in pairs.values() if len(pair) == 2]
    if deltas:
        low, high = bootstrap(deltas)
        summary["paired_text_image_minus_text_only"] = {"pairs": len(deltas), "mean_delta": sum(deltas) / len(deltas),
                                                       "percentile_ci95": [low, high], "seed": BOOTSTRAP_SEED,
                                                       "repetitions": BOOTSTRAP_REPETITIONS}
    summary["scope"] = ("Reverse-question relevance with a same-provider language model and a frozen multilingual MiniLM; "
                        "custom method, not the official Ragas implementation and not a human rating. Refusals are "
                        "reported separately (noncommittal) and are not set to zero.")
    summary["inputs_sha256"] = {"cases": hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                                "prompt": hashlib.sha256(PROMPT.encode()).hexdigest()}
    (output / "per_answer.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--live", action="store_true", help="send real API requests (dry run otherwise)")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.source)
    if args.analyze:
        print(json.dumps(analyze(cases, args.output), ensure_ascii=False, indent=2))
        return
    print(json.dumps(run(cases, args.output, args.live, args.budget_usd, args.max_requests), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
