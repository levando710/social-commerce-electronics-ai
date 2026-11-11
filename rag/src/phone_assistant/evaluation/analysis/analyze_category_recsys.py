"""Paired-user comparisons for a RecSys-only category benchmark (same pairs and bootstrap as Electronics)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.recommendation.recsys_data import ROOT, sha256

EXPERIMENT = ROOT / "artifacts/evaluation/results_amazon2023_baby_v1/experiment.json"
OUTPUT = ROOT / "artifacts/evaluation/results_amazon2023_baby_analysis_v1"
METRICS = ("hr@10", "ndcg@10")
# (first, second): the delta is second - first.  The first three are the Electronics pairs.
PAIRS = (("lightgcn", "vbpr"), ("graph_text_only", "graph_multimodal"),
         ("graph_multimodal", "graph_multimodal_gcl"), ("lightgcn", "graph_text_only"))
LIMITATIONS = ("Item-sampled 5-core subset of one category; one held-out target per user.",
               "Seeds are averaged per user before resampling; the interval describes users, not training randomness.",
               "A delta whose interval contains 0 is not evidence of a difference.",
               "RecSys-only: no queries, so retrieval and Co-Guiding are not evaluated on this benchmark.")


def parse_pairs(text):
    """``first:second,first:second`` -> ((first, second), ...); each delta is second - first."""
    pairs = tuple(tuple(part.strip() for part in item.split(":")) for item in text.split(",") if item.strip())
    if not pairs or any(len(pair) != 2 or not all(pair) for pair in pairs):
        raise ValueError("Pairs must look like first:second,first:second")
    return pairs


def analyze(experiment_path, output, pairs=PAIRS):
    if output.exists():
        raise FileExistsError("Preserve the first analysis; use a new output for a revised one")
    experiment = json.loads(Path(experiment_path).read_text(encoding="utf-8"))
    by_method = defaultdict(list)
    for run in experiment["runs"]:
        by_method[run["method"]].append(run)
    paired = {f"{second}_minus_{first}": {metric: paired_bootstrap(user_scores(by_method[first], metric),
                                                                   user_scores(by_method[second], metric))
                                          for metric in METRICS} for first, second in pairs}
    cold = {method: {"users": runs[0]["test"]["cold_item"]["users"],
                     "hr@10_mean": sum(run["test"]["cold_item"]["hr@10"] for run in runs) / len(runs)}
            for method, runs in sorted(by_method.items())}
    result = {"experiment_sha256": sha256(experiment_path), "source_sha256": sha256(Path(__file__)),
              "paired_bootstrap": paired, "cold_target_hr@10": cold,
              "popularity": {k: experiment["popularity"]["test"][k]["hr@10"] for k in ("all", "warm_item", "cold_item")},
              "limitations": list(LIMITATIONS)}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pairs", help="comparisons as first:second,first:second (default: the Baby/Electronics pairs)")
    args = parser.parse_args()
    pairs = parse_pairs(args.pairs) if args.pairs else PAIRS
    print(json.dumps(analyze(args.experiment, args.output, pairs), indent=2))


if __name__ == "__main__":
    main()
