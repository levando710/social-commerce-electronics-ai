from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts/models"
REPORT_DIR = PROJECT_ROOT / "artifacts/reports"
LOG_DIR = PROJECT_ROOT / "artifacts/logs"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_PHONE_DATA = RAW_DATA_DIR / "dataset_dien_thoai_15k.csv"
CLEAN_PHONE_DATA = PROCESSED_DATA_DIR / "phone_clean.csv"
SCRAPE_LOG = LOG_DIR / "scrape_log.csv"

PHONE_MODEL = MODEL_DIR / "phone_price_model.cbm"
PHONE_MODEL_LOWER = MODEL_DIR / "phone_price_model_lower.cbm"
PHONE_MODEL_MEDIAN = MODEL_DIR / "phone_price_model_median.cbm"
PHONE_MODEL_UPPER = MODEL_DIR / "phone_price_model_upper.cbm"
PHONE_MODEL_METADATA = MODEL_DIR / "phone_model_metadata.json"
PHONE_OPTUNA_DB = MODEL_DIR / "phone_optuna.db"

MODEL_COMPARISON_RESULTS = REPORT_DIR / "model_comparison_results.csv"
HIGH_CONFIDENCE_PREDICTIONS = REPORT_DIR / "high_confidence_predictions.csv"
MODEL_COMPARISON_CHART = REPORT_DIR / "model_comparison_chart.png"


def ensure_project_dirs():
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_utf8_output():
    import sys

    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
