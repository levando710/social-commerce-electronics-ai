import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SimilarProductMedianRegressor(BaseEstimator, RegressorMixin):
    """Median-price baseline with hierarchical fallbacks for sparse phone models."""

    def __init__(self):
        self.global_median_log_ = None
        self.lookup_tables_ = []

    @staticmethod
    def _prepare_frame(X):
        frame = X.copy()
        for col in ["brand", "model", "color", "origin", "lock_status", "warranty_type"]:
            if col in frame.columns:
                frame[col] = frame[col].fillna("unknown").astype(str).str.lower().str.strip()
        if "capacity_gb" in frame.columns:
            frame["capacity_bucket"] = frame["capacity_gb"].round().astype("Int64").astype(str)
        if "condition_score" in frame.columns:
            frame["condition_bucket"] = frame["condition_score"].fillna(-1).astype(int).astype(str)
        return frame

    def fit(self, X, y):
        frame = self._prepare_frame(X)
        frame["_target"] = np.asarray(y)
        self.global_median_log_ = float(np.median(frame["_target"]))

        key_levels = [
            ["model", "capacity_bucket", "condition_bucket", "lock_status"],
            ["model", "capacity_bucket", "condition_bucket"],
            ["model", "capacity_bucket"],
            ["model"],
            ["brand", "capacity_bucket", "condition_bucket", "lock_status"],
            ["brand", "capacity_bucket", "condition_bucket"],
            ["brand", "capacity_bucket"],
            ["brand"],
        ]

        self.lookup_tables_ = []
        for keys in key_levels:
            available_keys = [key for key in keys if key in frame.columns]
            if not available_keys:
                continue
            table = frame.groupby(available_keys)["_target"].median().to_dict()
            self.lookup_tables_.append((available_keys, table))
        return self

    def predict(self, X):
        frame = self._prepare_frame(X)
        preds = []
        for _, row in frame.iterrows():
            pred = self.global_median_log_
            for keys, table in self.lookup_tables_:
                lookup_key = tuple(row[key] for key in keys)
                if len(lookup_key) == 1:
                    lookup_key = lookup_key[0]
                if lookup_key in table:
                    pred = table[lookup_key]
                    break
            preds.append(pred)
        return np.asarray(preds)
