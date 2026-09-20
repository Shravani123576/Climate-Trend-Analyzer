from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt


def validate_input_data(df):
    """
    Validate the dataframe before anomaly detection.
    """

    required_columns = [
        "Year",
        "Temperature_Anomaly"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    if df[
        ["Year", "Temperature_Anomaly"]
    ].isnull().any().any():

        raise ValueError(
            "Input contains missing values."
        )

    return True


def calculate_z_scores(df):
    """
    Calculate the z-score for each temperature anomaly.
    """

    validate_input_data(df)

    data = df.copy()

    data["Z_Score"] = zscore(
        data["Temperature_Anomaly"]
    )

    return data


def detect_zscore_anomalies(
    df,
    threshold=2
):
    """
    Detect anomalies using the z-score method.

    A point is considered an anomaly when
    the absolute z-score is greater than or
    equal to the threshold.
    """

    data = calculate_z_scores(df)

    data["ZScore_Anomaly"] = (
        data["Z_Score"].abs() >= threshold
    )

    return data


def detect_iqr_anomalies(df):
    """
    Detect anomalies using the IQR method.
    """

    validate_input_data(df)

    data = df.copy()

    q1 = data[
        "Temperature_Anomaly"
    ].quantile(0.25)

    q3 = data[
        "Temperature_Anomaly"
    ].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)

    upper_bound = q3 + (1.5 * iqr)

    data["IQR_Anomaly"] = (
        (
            data["Temperature_Anomaly"]
            < lower_bound
        )
        |
        (
            data["Temperature_Anomaly"]
            > upper_bound
        )
    )

    return data


def compare_anomaly_methods(df):
    """
    Compare Z-score and IQR anomaly detection.
    """

    zscore_data = detect_zscore_anomalies(df)

    iqr_data = detect_iqr_anomalies(df)

    result = df[
        [
            "Year",
            "Temperature_Anomaly"
        ]
    ].copy()

    result["ZScore_Anomaly"] = (
        zscore_data["ZScore_Anomaly"]
    )

    result["IQR_Anomaly"] = (
        iqr_data["IQR_Anomaly"]
    )

    def classify(row):

        z = row["ZScore_Anomaly"]
        iqr = row["IQR_Anomaly"]

        if z and iqr:
            return "Both"

        if z and not iqr:
            return "Z-Score Only"

        if not z and iqr:
            return "IQR Only"

        return "None"

    result["Detection_Category"] = (
        result.apply(
            classify,
            axis=1
        )
    )

    return result


def get_anomaly_summary(df):
    """
    Generate a summary of anomaly counts.
    """

    comparison = compare_anomaly_methods(df)

    return {
        "ZScore_Anomalies": int(
            comparison["ZScore_Anomaly"].sum()
        ),
        "IQR_Anomalies": int(
            comparison["IQR_Anomaly"].sum()
        ),
        "Both_Methods": int(
            (
                comparison[
                    "Detection_Category"
                ]
                == "Both"
            ).sum()
        ),
        "ZScore_Only": int(
            (
                comparison[
                    "Detection_Category"
                ]
                == "Z-Score Only"
            ).sum()
        ),
        "IQR_Only": int(
            (
                comparison[
                    "Detection_Category"
                ]
                == "IQR Only"
            ).sum()
        )
    }


def create_anomaly_plot(
    df,
    output_path="outputs/graphs/anomaly_detection.png"
):
    """
    Create and save a temperature anomaly plot
    with Z-score detected anomalies highlighted.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    anomaly_data = detect_zscore_anomalies(
        df
    )

    normal = anomaly_data[
        ~anomaly_data["ZScore_Anomaly"]
    ]

    anomalies = anomaly_data[
        anomaly_data["ZScore_Anomaly"]
    ]

    plt.figure(figsize=(15, 6))

    plt.scatter(
        normal["Year"],
        normal["Temperature_Anomaly"],
        alpha=0.6,
        label="Normal"
    )

    plt.scatter(
        anomalies["Year"],
        anomalies["Temperature_Anomaly"],
        marker="x",
        s=70,
        label="Z-Score Anomaly"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        alpha=0.5
    )

    plt.title(
        "Global Temperature Anomaly Detection"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Temperature Anomaly (°C)"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def save_anomaly_results(
    df,
    output_path="outputs/tables/anomaly_detection_results.csv"
):
    """
    Save anomaly detection results to CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = compare_anomaly_methods(df)

    results.to_csv(
        output_path,
        index=False
    )

    return output_path


def save_anomaly_summary(
    df,
    output_path="outputs/tables/anomaly_summary.csv"
):
    """
    Save anomaly counts as a CSV table.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = get_anomaly_summary(df)

    summary_df = pd.DataFrame([
        summary
    ])

    summary_df.to_csv(
        output_path,
        index=False
    )

    return output_path


def run_anomaly_detection(
    input_path="data/processed/climate_trend_clean.csv"
):
    """
    Run the complete anomaly detection workflow.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    validate_input_data(df)

    results = compare_anomaly_methods(
        df
    )

    summary = get_anomaly_summary(
        df
    )

    create_anomaly_plot(
        df
    )

    save_anomaly_results(
        df
    )

    save_anomaly_summary(
        df
    )

    print("\n===== ANOMALY DETECTION =====")

    print(
        f"Z-Score anomalies : "
        f"{summary['ZScore_Anomalies']}"
    )

    print(
        f"IQR anomalies     : "
        f"{summary['IQR_Anomalies']}"
    )

    print(
        f"Detected by both  : "
        f"{summary['Both_Methods']}"
    )

    print(
        f"Z-Score only      : "
        f"{summary['ZScore_Only']}"
    )

    print(
        f"IQR only          : "
        f"{summary['IQR_Only']}"
    )

    print(
        "\nAnomaly detection completed successfully."
    )

    return {
        "results": results,
        "summary": summary
    }