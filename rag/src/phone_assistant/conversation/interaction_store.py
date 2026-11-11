"""Local demo visitor histories and an untrained metadata-based preference baseline.

Browser visitors have their own random token; never map them to Amazon user IDs.
Query logs are not automatically promoted to lasting interests. Explicit actions
are reversible and scored once per item/action, with a 30-day half-life.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import secrets
import sqlite3
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from phone_assistant.recommendation.recsys_data import metadata_text

ACTIONS = {"view", "like", "unlike", "save", "unsave", "query"}


class InteractionStore:
    def __init__(self, path, catalog_ids):
        self.path = Path(path)
        self.catalog_ids = set(catalog_ids)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS visitors(token TEXT PRIMARY KEY,created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                    visitor TEXT NOT NULL,event_id TEXT NOT NULL,kind TEXT NOT NULL,
                    item_id TEXT,query TEXT,created REAL NOT NULL,
                    PRIMARY KEY(visitor,event_id),FOREIGN KEY(visitor) REFERENCES visitors(token));
                CREATE INDEX IF NOT EXISTS visitor_event_time ON events(visitor,created DESC);
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_visitor(self):
        token = secrets.token_urlsafe(32)
        with self.connection() as connection:
            connection.execute("INSERT INTO visitors VALUES (?,?)", (token, time.time()))
        return token

    def exists(self, token):
        if not isinstance(token, str) or len(token) > 100:
            return False
        with self.connection() as connection:
            return connection.execute("SELECT 1 FROM visitors WHERE token=?", (token,)).fetchone() is not None

    def record(self, visitor, kind, *, item_id=None, query=None, event_id=None):
        if not isinstance(kind, str) or kind not in ACTIONS:
            raise ValueError("Unknown interaction type")
        if kind == "query":
            if not isinstance(query, str) or not query.strip() or len(query) > 2000 or item_id is not None:
                raise ValueError("Invalid query event")
        elif not isinstance(item_id, str) or item_id not in self.catalog_ids or query is not None:
            raise ValueError("Interaction must reference a catalog item")
        event_id = secrets.token_urlsafe(18) if event_id is None else event_id
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 100:
            raise ValueError("Invalid event ID")
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM visitors WHERE token=?", (visitor,)).fetchone():
                raise ValueError("Unknown visitor")
            previous = connection.execute("SELECT kind,item_id,query FROM events WHERE visitor=? AND event_id=?", (visitor, event_id)).fetchone()
            if previous:
                if previous != (kind, item_id, query):
                    raise ValueError("Event ID already used for a different interaction")
                return event_id
            try:
                connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                    (visitor, event_id, kind, item_id, query, time.time()))
            except sqlite3.IntegrityError:
                # A concurrent request with the same event ID won the insert; apply the same
                # idempotency rule as the sequential path instead of failing the retry.
                stored = connection.execute("SELECT kind,item_id,query FROM events WHERE visitor=? AND event_id=?",
                                            (visitor, event_id)).fetchone()
                if stored != (kind, item_id, query):
                    raise ValueError("Event ID already used for a different interaction") from None
        return event_id

    def profile(self, visitor, *, now=None):
        now = time.time() if now is None else now
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM visitors WHERE token=?", (visitor,)).fetchone():
                raise ValueError("Unknown visitor")
            events = connection.execute("SELECT kind,item_id,created FROM events WHERE visitor=? ORDER BY created DESC,rowid DESC LIMIT 2000", (visitor,)).fetchall()
            count = connection.execute("SELECT COUNT(*) FROM events WHERE visitor=?", (visitor,)).fetchone()[0]
        latest, weights, active = {}, {}, {}
        for kind, item_id, created in events:
            if kind == "query" or item_id not in self.catalog_ids:
                continue
            category = "like" if kind in ("like", "unlike") else "save" if kind in ("save", "unsave") else "view"
            latest.setdefault((item_id, category), (kind, created))
        for (item_id, category), (kind, created) in latest.items():
            active.setdefault(item_id, {})[category] = kind not in ("unlike", "unsave")
            if kind in ("unlike", "unsave"):
                continue
            weight = {"view": 1.0, "like": 3.0, "save": 5.0}[category]
            decay = 2 ** (-max(now - created, 0) / (30 * 86400))
            weights[item_id] = weights.get(item_id, 0.0) + weight * decay
        return {"event_count": count, "item_weights": weights, "active_actions": active,
                "history_window_events": 2000, "method": "metadata_tfidf_time_decay_untrained"}

    def reset(self, visitor):
        with self.connection() as connection:
            connection.execute("DELETE FROM events WHERE visitor=?", (visitor,))


class LocalPreferenceService:
    def __init__(self, documents, store):
        self.documents = sorted(documents, key=lambda doc: doc["doc_id"])
        self.ids = [doc["doc_id"] for doc in self.documents]
        self.indices = {doc_id: index for index, doc_id in enumerate(self.ids)}
        self.store = store
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, dtype=np.float32)
        self.vectors = self.vectorizer.fit_transform([metadata_text(doc) for doc in self.documents])

    def recommend(self, visitor, top_k=5):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        profile = self.store.profile(visitor)
        weights = profile["item_weights"]
        scores = np.asarray([float(doc.get("review_count") or 0) for doc in self.documents], dtype=np.float32)
        if weights:
            indices = [self.indices[item] for item in weights]
            values = np.asarray([weights[item] for item in weights], dtype=np.float32)
            mean = normalize(np.asarray(self.vectors[indices].multiply(values[:, None]).sum(axis=0)))
            scores = np.asarray(self.vectors @ mean.T).ravel()
        order = np.lexsort((np.arange(len(self.ids)), -scores))
        rows = []
        for index in order:
            doc = self.documents[index]
            if doc["doc_id"] in weights:
                continue
            rows.append({"doc_id": doc["doc_id"], "title": doc.get("title"), "brand": doc.get("brand"),
                         "score": float(scores[index]), "product_images": doc.get("product_image_paths", [])[:1]})
            if len(rows) == top_k:
                break
        return {"method": profile["method"] if weights else "catalog_review_count_prior",
                "personalized": bool(weights), "results": rows,
                "context": "this browser's recorded actions; metadata baseline, no trained user ID",
                "profile": profile}
