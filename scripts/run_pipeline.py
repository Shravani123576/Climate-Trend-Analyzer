from pathlib import Path
import logging

from src.data_loader import (
    load_raw_data
)

from src.preprocessing import (
    clean_climate_data,
    validate_clean_data,
    save_clean_data
)

from src.eda import (
    run_eda
)

from src.trend_analysis import (
    run_trend_analysis
)

from src.anomaly_detection import (
    run_anomaly_detection
)

from src.forecasting import (
    run_forecasting
)

from src.validation import (
    validate_model
)

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(
    __file__
).resolve().parent.parent

RAW_DATA = (
    BASE_DIR
    / "data"
    / "raw"
    / "GLB.Ts+dSST.csv"
)

PROCESSED_DATA = (
    BASE_DIR
    / "data"
    / "processed"
    / "climate_trend_clean.csv"
)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logging.basicConfig(
    filename=LOG_DIR / "pipeline.log",
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "ClimatePipeline"
)


def log_step(message):
    """
    Print and log a pipeline step.
    """

    print(message)

    logger.info(message)


def run_pipeline():
    """
    Run the complete Climate Trend Analyzer pipeline.
    """

    log_step(
        "\n===================================="
    )

    log_step(
        "CLIMATE TREND ANALYZER PIPELINE"
    )

    log_step(
        "===================================="
    )

    # --------------------------------------------------
    # STEP 1 — LOAD RAW DATA
    # --------------------------------------------------

    log_step(
        "\n[1/7] Loading raw NASA dataset..."
    )

    raw_df = load_raw_data(
        RAW_DATA
    )

    log_step(
        f"Raw dataset loaded: {raw_df.shape}"
    )

    # --------------------------------------------------
    # STEP 2 — PREPROCESS
    # --------------------------------------------------

    log_step(
        "\n[2/7] Cleaning and preprocessing..."
    )

    clean_df = clean_climate_data(
        raw_df
    )

    validate_clean_data(
        clean_df
    )

    save_clean_data(
        clean_df,
        PROCESSED_DATA
    )

    log_step(
        f"Clean dataset saved: {clean_df.shape}"
    )

    # --------------------------------------------------
    # STEP 3 — EDA
    # --------------------------------------------------

    log_step(
        "\n[3/7] Running exploratory data analysis..."
    )

    run_eda(
        PROCESSED_DATA
    )

    log_step(
        "EDA completed."
    )

    # --------------------------------------------------
    # STEP 4 — TREND ANALYSIS
    # --------------------------------------------------

    log_step(
        "\n[4/7] Running trend analysis..."
    )

    run_trend_analysis(
        PROCESSED_DATA
    )

    log_step(
        "Trend analysis completed."
    )

    # --------------------------------------------------
    # STEP 5 — ANOMALY DETECTION
    # --------------------------------------------------

    log_step(
        "\n[5/7] Running anomaly detection..."
    )

    run_anomaly_detection(
        PROCESSED_DATA
    )

    log_step(
        "Anomaly detection completed."
    )

    # --------------------------------------------------
    # STEP 6 — FORECASTING
    # --------------------------------------------------

    log_step(
        "\n[6/7] Running forecasting..."
    )

    forecast_results = run_forecasting(
        PROCESSED_DATA
    )

    log_step(
        "Forecasting completed."
    )

    # --------------------------------------------------
    # STEP 7 — VALIDATION
    # --------------------------------------------------

    log_step(
        "\n[7/7] Running model validation..."
    )

    arima_results = (
        forecast_results[
            "arima"
        ]
    )

    # The forecasting module evaluates ARIMA
    # against the held-out test set internally.
    #
    # Here we run the final reusable validation
    # layer against those predictions.

    validation_df = pd.read_csv(
        PROCESSED_DATA
    )

    total_rows = len(
        validation_df
    )

    # Recreate the same held-out test section
    # used by the forecasting workflow.

    test_years = 10

    test_actual = validation_df[
        "Temperature_Anomaly"
    ].iloc[
        -test_years:
    ]

    arima_predictions = (
        arima_results[
            "predictions"
        ]
    )

    validate_model(
        "ARIMA",
        test_actual,
        arima_predictions
    )

    log_step(
        "Validation completed."
    )

    # --------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------

    log_step(
        "\n===================================="
    )

    log_step(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    log_step(
        "====================================\n"
    )


if __name__ == "__main__":
    run_pipeline()