"""The result charts read their values from the report tables; check the parser and one full render."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag/src"))
sys.path.append(str(ROOT / "rag/artifacts/cache/report_tools"))
REPORT = ROOT / "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md"


class ResultChartsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.charts = importlib.import_module("phone_assistant.reporting.figures.generate_result_charts")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"plotting library not installed: {error}")

    def test_table_parser_reads_header_and_rows(self):
        lines = ["**Bảng 9.9. Mẫu.**", "", "| Tên | Giá trị |", "| --- | --- |", "| a | 0.5 |", "| b | — |", "", "hết"]
        header, rows = self.charts.table_after(lines, "**Bảng 9.9.")
        self.assertEqual(header, ["Tên", "Giá trị"])
        self.assertEqual([self.charts.number(row[1]) for row in rows], [0.5, None])

    def test_every_chart_renders_from_the_report(self):
        with tempfile.TemporaryDirectory() as folder:
            argv = sys.argv
            sys.argv = ["generate_result_charts", "--report", str(REPORT), "--output", folder]
            try:
                self.charts.main()
            finally:
                sys.argv = argv
            names = {path.name for path in Path(folder).glob("*.png")}
            self.assertEqual(len(names), 7)
            self.assertTrue((Path(folder) / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
