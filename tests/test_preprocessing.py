import pandas as pd

from src.preprocessing import (
    select_required_columns,
    rename_columns,
    handle_missing_values,
    convert_data_types,
    sort_by_year,
    clean_climate_data,
    validate_clean_data,
)


def create_sample_data():
    return pd.DataFrame({
        "Year": [1882, 1880, 1881],
        "J-D": [0.15, -0.19, -0.08]
    })


def test_select_required_columns():

    df = pd.DataFrame({
        "Year": [1880],
        "J-D": [-0.19],
        "Jan": [-0.19]
    })

    result = select_required_columns(df)

    assert list(result.columns) == ["Year", "J-D"]


def test_rename_columns():

    df = create_sample_data()

    result = rename_columns(df)

    assert "Temperature_Anomaly" in result.columns
    assert "J-D" not in result.columns


def test_handle_missing_values():

    df = pd.DataFrame({
        "Year": [1880, 1881, 1882],
        "Temperature_Anomaly": [-0.19, None, 0.15]
    })

    result = handle_missing_values(df)

    assert result["Temperature_Anomaly"].isnull().sum() == 0
    assert len(result) == 2


def test_convert_data_types():

    df = pd.DataFrame({
        "Year": ["1880", "1881"],
        "Temperature_Anomaly": ["-0.19", "0.05"]
    })

    result = convert_data_types(df)

    assert pd.api.types.is_numeric_dtype(
        result["Year"]
    )

    assert pd.api.types.is_numeric_dtype(
        result["Temperature_Anomaly"]
    )


def test_sort_by_year():

    df = create_sample_data()

    result = sort_by_year(df)

    assert result["Year"].tolist() == [1880, 1881, 1882]


def test_clean_climate_data():

    df = create_sample_data()

    result = clean_climate_data(df)

    assert list(result.columns) == [
        "Year",
        "Temperature_Anomaly"
    ]

    assert result["Year"].tolist() == [
        1880,
        1881,
        1882
    ]


def test_validate_clean_data():

    df = create_sample_data()

    df = clean_climate_data(df)

    assert validate_clean_data(df) is True