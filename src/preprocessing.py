from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["Year", "J-D"]


def select_required_columns(df):
    """
    Select the columns required for climate trend analysis.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    return df[REQUIRED_COLUMNS].copy()


def rename_columns(df):
    """
    Rename NASA's J-D annual temperature anomaly column
    to a more descriptive project name.
    """

    return df.rename(
        columns={
            "J-D": "Temperature_Anomaly"
        }
    )


def handle_missing_values(df):
    """
    Remove rows where the temperature anomaly is missing.
    """

    data = df.copy()

    data = data.dropna(
        subset=["Year", "Temperature_Anomaly"]
    )

    return data


def convert_data_types(df):
    """
    Convert columns to appropriate numeric data types.
    """

    data = df.copy()

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["Temperature_Anomaly"] = pd.to_numeric(
        data["Temperature_Anomaly"],
        errors="coerce"
    )

    return data


def sort_by_year(df):
    """
    Sort the dataset chronologically.
    """

    data = df.copy()

    return data.sort_values(
        by="Year"
    ).reset_index(drop=True)


def clean_climate_data(df):
    """
    Complete preprocessing pipeline for the climate dataset.
    """

    data = select_required_columns(df)

    data = rename_columns(data)

    data = convert_data_types(data)

    data = handle_missing_values(data)

    data = sort_by_year(data)

    return data


def validate_clean_data(df):
    """
    Validate the final cleaned dataset.
    """

    required_columns = [
        "Year",
        "Temperature_Anomaly"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    if df.empty:
        raise ValueError(
            "Cleaned dataset is empty."
        )

    if df["Year"].isnull().any():
        raise ValueError(
            "Year contains missing values."
        )

    if df["Temperature_Anomaly"].isnull().any():
        raise ValueError(
            "Temperature_Anomaly contains missing values."
        )

    if not df["Year"].is_monotonic_increasing:
        raise ValueError(
            "Year values are not sorted."
        )

    return True


def save_clean_data(df, output_path):
    """
    Save the cleaned dataset to the processed-data directory.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    return output_path