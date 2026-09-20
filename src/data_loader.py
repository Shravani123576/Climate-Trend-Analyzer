from pathlib import Path
import pandas as pd


def load_raw_data(file_path):
    """
    Load the raw NASA climate dataset.

    The NASA CSV contains a title line before the actual
    column header, so the first row is skipped.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(
        file_path,
        skiprows=1,
        na_values="***"
    )

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    return df


def dataset_summary(df):
    """
    Return basic information about the dataset.
    """

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Column Names": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict()
    }


def validate_dataset(df):
    """
    Validate that the dataset is not empty.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    return True


def preview_data(df, rows=5):
    """
    Return the first few rows of the dataset.
    """

    return df.head(rows)