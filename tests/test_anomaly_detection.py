import pandas as pd
import numpy as np

from src.anomaly_detection import (
    calculate_z_scores,
    detect_zscore_anomalies,
    detect_iqr_anomalies,
    compare_anomaly_methods,
    get_anomaly_summary,
    validate_input_data,
)


def create_sample_data():

    return pd.DataFrame({
        "Year": [
            2000,
            2001,
            2002,
            2003,
            2004,
            2005,
            2006,
            2007,
            2008,
            2009
        ],
        "Temperature_Anomaly": [
            0.10,
            0.12,
            0.15,
            0.13,
            0.14,
            0.11,
            0.16,
            0.15,
            0.14,
            2.00
        ]
    })


def test_validate_input_data():

    df = create_sample_data()

    assert validate_input_data(df) is True


def test_calculate_z_scores():

    df = create_sample_data()

    result = calculate_z_scores(df)

    assert "Z_Score" in result.columns

    assert len(result) == len(df)

    assert result["Z_Score"].notna().all()


def test_detect_zscore_anomalies():

    df = create_sample_data()

    result = detect_zscore_anomalies(
        df,
        threshold=1.5
    )

    assert "ZScore_Anomaly" in result.columns

    assert result["ZScore_Anomaly"].dtype == bool


def test_detect_iqr_anomalies():

    df = create_sample_data()

    result = detect_iqr_anomalies(df)

    assert "IQR_Anomaly" in result.columns

    assert result["IQR_Anomaly"].dtype == bool


def test_compare_anomaly_methods():

    df = create_sample_data()

    result = compare_anomaly_methods(df)

    assert "ZScore_Anomaly" in result.columns

    assert "IQR_Anomaly" in result.columns

    assert "Detection_Category" in result.columns

    assert len(result) == len(df)


def test_anomaly_summary():

    df = create_sample_data()

    summary = get_anomaly_summary(df)

    assert "ZScore_Anomalies" in summary

    assert "IQR_Anomalies" in summary

    assert "Both_Methods" in summary

    assert "ZScore_Only" in summary

    assert "IQR_Only" in summary


def test_anomaly_summary_counts():

    df = create_sample_data()

    summary = get_anomaly_summary(df)

    total_anomalies = (
        summary["Both_Methods"]
        + summary["ZScore_Only"]
        + summary["IQR_Only"]
    )

    total_detected_by_methods = (
        summary["ZScore_Anomalies"]
        + summary["IQR_Anomalies"]
    )

    assert total_anomalies >= 0

    assert total_detected_by_methods >= 0

    assert (
        total_anomalies
        <= len(df)
    )