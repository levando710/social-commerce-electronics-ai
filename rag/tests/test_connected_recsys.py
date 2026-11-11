from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.preparation.prepare_connected_recsys import choose_support, connect_cache, import_catalog_rows


def row(user, item, ts, rating=5):
    return {"user_id": user, "parent_asin": item, "timestamp": str(ts), "rating": str(rating)}


class ConnectedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.connection = connect_cache(Path(self.temp.name) / "events.sqlite")

    def tearDown(self):
        self.connection.close()
        self.temp.cleanup()

    def test_all_source_rows_for_protected_user_are_rejected(self):
        protected = hashlib.sha256(b"protected").hexdigest()
        audit = import_catalog_rows(self.connection,
            [row("protected", "a", 1), row("protected", "b", 999), row("other", "a", 2)],
            {"a", "b"}, {protected})
        self.assertEqual(audit["excluded_protected_user_rows"], 2)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM events WHERE user_id=?", (protected,)).fetchone()[0], 0)

    def test_latest_rating_and_conservative_tie_before_positive_filter(self):
        import_catalog_rows(self.connection, [row("u", "a", 1), row("u", "a", 2, 1),
            row("u", "b", 2), row("u", "b", 2, 2), row("u", "c", 3)], {"a", "b", "c"}, set())
        events, users = choose_support(self.connection, threshold=4, min_items=1, max_users=5)
        self.assertEqual([e["item_id"] for e in events], ["c"])
        self.assertEqual(len(users), 1)

    def test_catalog_filter_and_user_limit_are_deterministic(self):
        records = [row(user, item, ts) for user in ("u", "v", "w") for ts, item in enumerate(("a", "b", "outside"), 1)]
        import_catalog_rows(self.connection, reversed(records), {"a", "b"}, set())
        events, users = choose_support(self.connection, threshold=4, min_items=2, max_users=2)
        self.assertEqual(len(users), 2)
        self.assertEqual(len(events), 4)
        self.assertEqual(users, sorted(users))
        self.assertNotIn("outside", {e["item_id"] for e in events})


if __name__ == "__main__":
    unittest.main()
