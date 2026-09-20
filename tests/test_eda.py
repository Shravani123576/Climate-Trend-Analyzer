import pandas as pd

from src.eda import (
    calculate_summary_statistics,
    create_summary_table,
    calculate_rolling_average,
    get_top_hottest_years,
    get_top_coldest_years,
    load_clean_data,
)


def create_sample_data():

    return pd.DataFrame({
        "Year": [1880, 1881, 1882, 1883, 1884],
        "Temperature_Anomaly": [
            -0.19,
            -0.16,
            0.15,
            -0.30,
            -0.13
        ]
    })


def test_summary_statistics():

    df = create_sample_data()

    summary = calculate_summary_statistics(df)

    assert "Mean" in summary
    assert "Median" in summary
    assert "Minimum" in summary
    assert "Maximum" in summary

    assert summary["Minimum"] == -0.30
    assert summary["Maximum"] == 0.15


def test_summary_table():

    df = create_sample_data()

    table = create_summary_table(df)

    assert len(table) == 1
    assert "Mean" in table.columns
    assert "Maximum" in table.columns


def test_rolling_average():

    df = create_sample_data()

    result = calculate_rolling_average(
        df,
        window=3
    )

    assert "Rolling_Avg" in result.columns

    assert pd.isna(
        result.loc[0, "Rolling_Avg"]
    )

    assert pd.isna(
        result.loc[1, "Rolling_Avg"]
    )

    assert not pd.isna(
        result.loc[2, "Rolling_Avg"]
    )


def test_top_hottest_years():

    df = create_sample_data()

    result = get_top_hottest_years(
        df,
        n=2
    )

    assert len(result) == 2
    assert result.iloc[0]["Year"] == 1882


def test_top_coldest_years():

    df = create_sample_data()

    result = get_top_coldest_years(
        df,
        n=2
    )

    assert len(result) == 2
    assert result.iloc[0]["Year"] == 1883


def test_load_clean_data(tmp_path):

    df = create_sample_data()

    test_file = tmp_path / "climate_test.csv"

    df.to_csv(
        test_file,
        index=False
    )

    result = load_clean_data(
        test_file
    )

    assert not result.empty
    assert list(result.columns) == [
        "Year",
        "Temperature_Anomaly"
    ]