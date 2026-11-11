from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.preparation import prepare_mind_recsys as mind


def row(impression, user, stamp, history, shown):
    return "\t".join([str(impression), user, stamp, " ".join(history), " ".join(shown)]) + "\n"


class MindPreparationTests(unittest.TestCase):
    def test_time_parsing_orders_am_pm_correctly(self):
        self.assertLess(mind.parse_time("11/12/2019 11:59:59 AM"), mind.parse_time("11/12/2019 12:00:01 PM"))
        self.assertLess(mind.parse_time("11/09/2019 9:00:00 PM"), mind.parse_time("11/10/2019 1:00:00 AM"))

    def test_targets_are_new_clicks_never_history_and_history_is_training_only(self):
        known = {f"N{n}" for n in range(1, 12)}
        record = {"history": ["N1", "N2"], "clicks": [(10, "N3"), (20, "N4"), (30, "N2"), (40, "N5"), (50, "N6")]}
        history, early, validation, test = mind.user_events(record, known)
        self.assertEqual(history, ["N1", "N2"])
        self.assertEqual([item for _, item in early], ["N3", "N4"])  # N2 is history, so it is not a new click
        self.assertEqual((validation[1], test[1]), ("N5", "N6"))

    def test_user_with_too_few_new_clicks_is_dropped_and_unknown_items_ignored(self):
        known = {"N1", "N2", "N3"}
        self.assertIsNone(mind.user_events({"history": [], "clicks": [(1, "N1"), (2, "N2")]}, known))
        self.assertIsNone(mind.user_events({"history": [], "clicks": [(1, "N1"), (2, "N2"), (3, "NX")]}, known))

    def test_end_to_end_split_loads_with_the_frozen_loader_and_has_no_overlap(self):
        from phone_assistant.recommendation.recsys_data import InteractionData
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "src"
            source.mkdir()
            items = [f"N{n}" for n in range(1, 13)]
            (source / "news.tsv").write_text("".join(f"{i}\tcat\tsub\tTitle {i}\tAbstract {i}\n" for i in items), encoding="utf-8")
            lines = ""
            for user in ("U1", "U2", "U3"):
                lines += row(1, user, "11/10/2019 9:00:00 AM", ["N1", "N2"], [f"{i}-1" for i in items[2:8]] + ["N9-0"])
                lines += row(2, user, "11/11/2019 9:00:00 AM", ["N1", "N2"], ["N10-1", "N11-0"])
            (source / "behaviors.tsv").write_text(lines, encoding="utf-8")
            summary = mind.prepare(source, Path(folder) / "out", sampled_users=10, seed=1)
            data = InteractionData(Path(folder) / "out")
            self.assertEqual(data.n_users, 3)
            self.assertEqual(summary["train_history_edges"], 6)
            self.assertEqual(len(data.pairs["test"]), 3)
            with self.assertRaises(FileExistsError):
                mind.prepare(source, Path(folder) / "out")


if __name__ == "__main__":
    unittest.main()
