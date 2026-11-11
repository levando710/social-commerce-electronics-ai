"""Query a trained recommender; no user index gives the cold-start Pop fallback."""
import argparse
import json
from pathlib import Path

from phone_assistant.recommendation.recsys_data import ROOT
from phone_assistant.recommendation.recsys_models import METHODS
from phone_assistant.recommendation.recsys_service import RecommendationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-index", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", choices=METHODS, default="graph_multimodal_gcl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/fixtures/recommendation')
    parser.add_argument("--models", type=Path, default=ROOT / 'artifacts/models/recommendation')
    parser.add_argument("--experiment", type=Path, default=ROOT / 'artifacts/evaluation/results_recsys/experiment.json')
    args = parser.parse_args()
    service = RecommendationService(split=args.split, models=args.models, experiment=args.experiment,
                                     method=args.method, seed=args.seed)
    print(json.dumps(service.recommend(args.user_index, args.top_k), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
