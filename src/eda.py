from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_clean_data(file_path):
    """
    Load the already-cleaned climate dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to the processed climate CSV.

    Returns
    -------
    pandas.DataFrame
        Clean climate dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

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

    return df


def calculate_summary_statistics(df):
    """
    Calculate descriptive statistics for temperature anomalies.
    """

    temperature = df["Temperature_Anomaly"]

    return {
        "Mean": temperature.mean(),
        "Median": temperature.median(),
        "Minimum": temperature.min(),
        "Maximum": temperature.max(),
        "Standard_Deviation": temperature.std(),
        "Count": temperature.count()
    }


def create_summary_table(df):
    """
    Convert summary statistics into a DataFrame.
    """

    summary = calculate_summary_statistics(df)

    return pd.DataFrame([summary])


def create_temperature_distribution(
    df,
    output_path="outputs/graphs/temperature_distribution.png"
):
    """
    Create and save a temperature anomaly distribution plot.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["Temperature_Anomaly"].dropna(),
        bins=25,
        kde=True
    )

    plt.title(
        "Distribution of Global Temperature Anomalies"
    )

    plt.xlabel(
        "Temperature Anomaly (°C)"
    )

    plt.ylabel(
        "Number of Years"
    )

    plt.grid(
        alpha=0.3
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_temperature_boxplot(
    df,
    output_path="outputs/graphs/boxplot_temperature.png"
):
    """
    Create and save a boxplot of temperature anomalies.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 3))

    plt.boxplot(
        df["Temperature_Anomaly"].dropna(),
        vert=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.5
    )

    plt.title(
        "Global Temperature Anomaly Box Plot"
    )

    plt.xlabel(
        "Temperature Anomaly (°C)"
    )

    plt.grid(
        alpha=0.3
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_temperature_trend(
    df,
    output_path="outputs/graphs/temperature_trend.png"
):
    """
    Create and save the yearly temperature anomaly trend.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        df["Year"],
        df["Temperature_Anomaly"],
        linewidth=2
    )

    plt.title(
        "Global Temperature Anomaly Over Time"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Temperature Anomaly (°C)"
    )

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def calculate_rolling_average(
    df,
    window=10
):
    """
    Calculate a rolling average of temperature anomalies.
    """

    data = df.copy()

    data["Rolling_Avg"] = (
        data["Temperature_Anomaly"]
        .rolling(window=window)
        .mean()
    )

    return data


def create_rolling_average_plot(
    df,
    window=10,
    output_path="outputs/graphs/rolling_average.png"
):
    """
    Create and save the original series together with
    its rolling average.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = calculate_rolling_average(
        df,
        window=window
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        data["Year"],
        data["Temperature_Anomaly"],
        alpha=0.4,
        label="Original"
    )

    plt.plot(
        data["Year"],
        data["Rolling_Avg"],
        linewidth=3,
        label=f"{window}-Year Rolling Average"
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def get_top_hottest_years(
    df,
    n=10
):
    """
    Return the years with the highest temperature anomalies.
    """

    return (
        df.sort_values(
            "Temperature_Anomaly",
            ascending=False
        )
        .head(n)
        .copy()
    )


def get_top_coldest_years(
    df,
    n=10
):
    """
    Return the years with the lowest temperature anomalies.
    """

    return (
        df.sort_values(
            "Temperature_Anomaly",
            ascending=True
        )
        .head(n)
        .copy()
    )


def save_eda_tables(
    df,
    output_dir="outputs/tables"
):
    """
    Save EDA summary tables.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = create_summary_table(df)

    summary.to_csv(
        output_dir / "summary_statistics.csv",
        index=False
    )

    hottest = get_top_hottest_years(df)

    hottest.to_csv(
        output_dir / "top10_hottest_years.csv",
        index=False
    )

    coldest = get_top_coldest_years(df)

    coldest.to_csv(
        output_dir / "top10_coldest_years.csv",
        index=False
    )


def run_eda(
    input_path="data/processed/climate_trend_clean.csv"
):
    """
    Execute the complete EDA workflow.
    """

    df = load_clean_data(input_path)

    print("\n===== DATASET INFORMATION =====")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Columns : {df.columns.tolist()}")

    print("\n===== SUMMARY STATISTICS =====")

    summary = calculate_summary_statistics(df)

    for key, value in summary.items():
        if key == "Count":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    print("\n===== EXTREME YEARS =====")

    hottest = get_top_hottest_years(df)

    coldest = get_top_coldest_years(df)

    print(
        f"Warmest Year: "
        f"{hottest.iloc[0]['Year']}"
    )

    print(
        f"Coldest Year: "
        f"{coldest.iloc[0]['Year']}"
    )

    save_eda_tables(df)

    create_temperature_distribution(df)

    create_temperature_boxplot(df)

    create_temperature_trend(df)

    create_rolling_average_plot(df)

    print("\nEDA completed successfully.")

    return df