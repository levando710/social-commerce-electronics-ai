"""Summarize complete and failed RAG requests without converting judge votes into human truth."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from phone_assistant.evaluation.metrics.answer_evaluation import summarize_claims
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256
from phone_assistant.training.train_recsys_portable import atomic_json

SOURCE = ROOT / 'artifacts/evaluation/results_rag_generation_pilot_v1'
OUTPUT = ROOT / 'artifacts/evaluation/results_rag_generation_analysis_v1'
ARMS = ("text_only", "text_image")
QUESTION_TYPES = ("technical", "visual")
MANUAL_COLUMNS = ("human_faithfulness", "human_relevance_0_1_2", "human_visual_support", "reviewer")


def aggregate(rows):
    valid = [row for row in rows if row["judgment"]["status"] == "judged"]
    claims = [claim for row in valid for claim in row["judgment"]["claims"]]
    faithfulness = [row["judgment"]["metrics"]["faithfulness"] for row in valid if row["judgment"]["metrics"]["faithfulness"] is not None]
    return {"planned": len(rows), "generated": sum(bool(row["answer"]) for row in rows), "judged": len(valid),
        "generated_fraction": sum(bool(row["answer"]) for row in rows) / len(rows) if rows else None,
        "judged_fraction_of_planned": len(valid) / len(rows) if rows else None,
        "claim_micro": summarize_claims(claims), "answer_macro_faithfulness": float(np.mean(faithfulness)) if faithfulness else None,
        "answer_macro_relevance_0_to_2": float(np.mean([row["judgment"]["relevance"] for row in valid])) if valid else None,
        "image_supported_claims": sum(claim["supported"] is True and claim["image_reference_valid"] for claim in claims)}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve the first pilot analysis")
    records = read_jsonl(SOURCE / "records.jsonl")
    summary = json.loads((SOURCE / "summary.json").read_text())
    if sha256(SOURCE / "records.jsonl") != summary["records_sha256"]:
        raise ValueError("Saved pilot records changed")
    for row in records:
        for image in row["images"]:
            if sha256(Path(image["path"])) != image["sha256"]:
                raise ValueError("Image evidence changed")
    by_arm = {arm: aggregate([r for r in records if r["arm"] == arm]) for arm in ARMS}
    by_type = {kind: {arm: aggregate([r for r in records if r["arm"] == arm and r["question_type"] == kind]) for arm in ARMS} for kind in QUESTION_TYPES}
    pairs = defaultdict(dict)
    for row in records:
        pairs[row["source_id"], row["question_type"]][row["arm"]] = row
    complete_pairs = []
    for (source_id, kind), arms in pairs.items():
        if set(arms) == set(ARMS) and all(r["judgment"]["status"] == "judged" for r in arms.values()):
            complete_pairs.append({"source_id": source_id, "question_type": kind,
                "text_only": arms["text_only"]["judgment"]["metrics"], "text_image": arms["text_image"]["judgment"]["metrics"]})
    flagged = [{"case_id": row["case_id"], "claim": claim} for row in records if row["judgment"]["status"] == "judged"
        for claim in row["judgment"]["claims"] if claim["supported"] is not True]
    manual_path = SOURCE / "manual_answer_labels.csv"
    with manual_path.open(encoding="utf-8-sig", newline="") as handle:
        manual = list(csv.DictReader(handle))
    result = {"records_sha256": sha256(SOURCE / "records.jsonl"), "source_sha256": sha256(Path(__file__)),
        "request_journal_sha256": sha256(SOURCE / "requests.jsonl"), "requests_started": summary["requests_started"],
        "overall": aggregate(records), "by_arm": by_arm, "by_question_type": by_type,
        "complete_judged_pairs": complete_pairs, "flagged_claims": flagged,
        "manual_rows": len(manual), "manual_cells_filled_at_audit": sum(bool(row[column].strip()) for row in manual for column in MANUAL_COLUMNS),
        "limitations": ["Nine generated and eight judged answers out of twelve planned; keep missing and failed cases in coverage.",
            "Only two of three source groups have both arms judged; no independent user interval.",
            "Judge unsupported flags include statements about missing evidence; these may be judge errors and require human adjudication.",
            "Do not equate same-provider judge faithfulness with factual correctness or a real-world hallucination rate.",
            "Reference/quote validation establishes matching supplied sources, not semantic entailment.",
            "No automatic retries after quota exhaustion; raw outputs and original judgments remain unchanged."]}
    OUTPUT.mkdir(parents=True); atomic_json(OUTPUT / "analysis.json", result)
    (OUTPUT / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Pilot sinh RAG có/không ảnh — kết quả và giới hạn", "",
        f"Dự kiến {len(records)} câu trả lời; sinh được {result['overall']['generated']}, chấm tự động được {result['overall']['judged']}. Đã bắt đầu {summary['requests_started']} request; dừng khi Gemini báo 429/hết quota ngày, không retry. Một probe kết nối trước pilot được ghi riêng.", "",
        "| Arm | Dự kiến | Sinh được | Judge hợp lệ | Supported / unsupported / unknown | Faithfulness theo judge* |", "|---|---:|---:|---:|---|---:|"]
    for arm, row in by_arm.items():
        claim = row["claim_micro"]; value = claim["faithfulness"]
        lines.append(f"| {arm} | {row['planned']} | {row['generated']} | {row['judged']} | {claim['supported']} / {claim['unsupported']} / {claim['unknown']} | {value:.6f} |")
    lines += ["", "*Tỷ lệ supported trên claim được judge xác định, không tính unknown vào mẫu số; không phải độ chính xác do con người xác nhận. Relevance 0–2 do cùng provider chấm đạt trung bình 2 trên các output hợp lệ, chưa xác nhận bằng người chấm.", "",
        f"Có {len(complete_pairs)} cặp câu hỏi hoàn chỉnh, thuộc {len({r['source_id'] for r in complete_pairs})} nhóm context; chưa kết luận lợi ích của ảnh từ mẫu nhỏ và các giá trị trên.", "",
        "## Trường hợp cần xem lại", "",
        "Judge gắn unsupported cho các câu nói không có thông tin/ảnh để xác định vị trí camera. Ví dụ A51 là context thứ ba, trong khi generator chỉ được gửi ảnh của S20+ và M51; phát biểu không có ảnh A51 cần được người chấm kiểm lại thay vì mặc nhận là hallucination. Không sửa nhãn judge gốc để làm đẹp điểm.", "",
        "AI đã xem hai ảnh S20+/M51 để đối chiếu: ảnh thứ nhất có mặt trước và mặt sau, ảnh thứ hai là mặt trước. Đây là ghi nhận kiểm tra bằng AI, không phải nhãn người hoặc đánh giá toàn bộ claim.", "",
        "## Chấm bởi người thật", "",
        "Mở `../results_rag_generation_pilot_v1/human_review.html`, đọc đúng evidence và điền `manual_answer_labels.csv` trong cùng thư mục. Không lấy điểm judge làm đáp án chuẩn. Cột nhãn người được giữ riêng, không tự điền.", ""]
    lines += result["limitations"]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "complete_pairs": len(complete_pairs), "manual_cells_filled": result["manual_cells_filled_at_audit"]}, indent=2))


if __name__ == "__main__":
    main()
