"""Re-measure the pricing model with a leak-free early-stopping protocol (new output, frozen model untouched).

``phonetrain`` passes the test split as ``eval_set`` while early stopping, so the number of trees is
chosen using the test data.  This command keeps the identical test split (same seed), carves a
validation split out of the *training* rows for early stopping, and reports test error that was
never used for any choice.  It first checks that the frozen model reproduces its stored MAPE on the
same test rows, which proves the split is the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from phone_pricing.common.project_paths import CLEAN_PHONE_DATA, PHONE_MODEL, PHONE_MODEL_METADATA, REPORT_DIR
from phone_pricing.features.phone_features import CAT_FEATURES, FEATURES

OUTPUT = REPORT_DIR / "holdout_protocol_v1"
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.1
SPLIT_SEED = 42
REPEAT_SEEDS = (42, 43, 44)
EARLY_STOPPING_ROUNDS = 50
BASELINE_KEYS = ("model", "capacity_gb")


def load_frame(path):
    frame = pd.read_csv(path)
    frame = frame[frame["price_outlier"] == 0].copy()
    frame[CAT_FEATURES] = frame[CAT_FEATURES].fillna("unknown")
    for column in CAT_FEATURES:
        frame[column] = frame[column].astype(str).str.lower().str.strip()
    return frame


def split_three_way(frame, validation_seed):
    """Test split identical to ``phonetrain``; validation is taken only from its training part."""
    train_all, test = train_test_split(frame, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    train, validation = train_test_split(train_all, test_size=VALIDATION_FRACTION, random_state=validation_seed)
    return train, validation, test


def metrics(log_true, log_pred):
    true, predicted = np.expm1(log_true), np.expm1(log_pred)
    return {"mape_percent": float(mean_absolute_percentage_error(true, predicted) * 100),
            "median_ape_percent": float(np.median(np.abs(true - predicted) / true) * 100),
            "rmse_vnd": float(root_mean_squared_error(true, predicted))}


def median_baseline(train, test):
    """Median listing price of the same model and capacity, falling back to the global median."""
    table = train.groupby(list(BASELINE_KEYS))["log_price"].median()
    fallback = train["log_price"].median()
    keys = list(zip(*(test[key] for key in BASELINE_KEYS)))
    return np.array([table.get(key, fallback) for key in keys])


def run(data_path, output, params):
    if output.exists():
        raise FileExistsError("Preserve the first run; use a new output to repeat it")
    frame = load_frame(data_path)
    _, _, test = split_three_way(frame, SPLIT_SEED)
    frozen = CatBoostRegressor()
    frozen.load_model(str(PHONE_MODEL))
    stored = json.loads(PHONE_MODEL_METADATA.read_text(encoding="utf-8"))
    reproduced = metrics(test["log_price"], frozen.predict(test[FEATURES]))
    if abs(reproduced["mape_percent"] - stored["mape_percent"]) > 1e-6:
        raise ValueError("Frozen model does not reproduce its stored MAPE on this split; the test rows differ")
    repeats = []
    for seed in REPEAT_SEEDS:
        train, validation, repeat_test = split_three_way(frame, seed)
        if not repeat_test.index.equals(test.index):
            raise ValueError("Test rows changed between repeats")
        if set(validation.index) & set(test.index):
            raise ValueError("Validation overlaps the test split")
        model = CatBoostRegressor(**params, cat_features=CAT_FEATURES, verbose=False, random_seed=seed,
                                  loss_function="RMSE", eval_metric="RMSE", allow_writing_files=False)
        model.fit(train[FEATURES], train["log_price"], eval_set=(validation[FEATURES], validation["log_price"]),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        repeats.append({"seed": seed, "train_rows": len(train), "validation_rows": len(validation),
                        "trees_kept": int(model.tree_count_), "test": metrics(test["log_price"], model.predict(test[FEATURES]))})
        print(repeats[-1], flush=True)
    train_all = frame.drop(index=test.index)
    summary = {key: {"mean": float(np.mean([r["test"][key] for r in repeats])),
                     "sd": float(np.std([r["test"][key] for r in repeats]))} for key in repeats[0]["test"]}
    result = {"data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(), "test_rows": len(test),
              "protocol": {"test_fraction": TEST_FRACTION, "validation_fraction_of_train": VALIDATION_FRACTION,
                           "split_seed": SPLIT_SEED, "repeat_seeds": list(REPEAT_SEEDS),
                           "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "params_source": "frozen model metadata"},
              "frozen_model_same_test_rows": {**reproduced, "trees": int(frozen.tree_count_),
                                              "note": "Early stopping used these test rows; optimistic by construction."},
              "leak_free_catboost": {"repeats": repeats, "summary": summary},
              "median_model_capacity_baseline": metrics(test["log_price"], median_baseline(train_all, test)),
              "limitations": ["Hyper-parameters came from an earlier tuning run that also saw this split; the test rows are not untouched for tuning.",
                              "Labels are listing prices, not transaction prices.",
                              "Outlier flag was computed on the full dataset before splitting.",
                              "Only ad_id deduplication was applied; reposted near-duplicate listings may span train and test."]}
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CLEAN_PHONE_DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    params = json.loads(PHONE_MODEL_METADATA.read_text(encoding="utf-8"))["params"]
    result = run(args.data, args.output, params)
    print(json.dumps({key: result[key] for key in ("frozen_model_same_test_rows", "median_model_capacity_baseline")}, indent=2))
    print(json.dumps(result["leak_free_catboost"]["summary"], indent=2))


if __name__ == "__main__":
    main()
