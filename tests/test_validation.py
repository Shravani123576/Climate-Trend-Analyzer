import numpy as np
import pandas as pd

from src.validation import (
    validate_arrays,
    calculate_regression_metrics,
    calculate_residuals,
    summarize_residuals,
    create_validation_table,
)


def test_validate_arrays():

    actual = np.array([
        1.0,
        2.0,
        3.0
    ])

    predicted = np.array([
        1.1,
        1.9,
        3.2
    ])

    y_true, y_pred = validate_arrays(
        actual,
        predicted
    )

    assert len(y_true) == 3
    assert len(y_pred) == 3


def test_validate_arrays_length_error():

    actual = np.array([
        1.0,
        2.0,
        3.0
    ])

    predicted = np.array([
        1.0,
        2.0
    ])

    try:
        validate_arrays(
            actual,
            predicted
        )
        assert False
    except ValueError:
        assert True


def test_regression_metrics():

    actual = np.array([
        1.0,
        2.0,
        3.0
    ])

    predicted = np.array([
        1.0,
        2.0,
        3.0
    ])

    metrics = calculate_regression_metrics(
        actual,
        predicted
    )

    assert metrics["MSE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["R2"] == 1


def test_residuals():

    actual = np.array([
        1.0,
        2.0,
        3.0
    ])

    predicted = np.array([
        0.5,
        2.5,
        3.0
    ])

    residuals = calculate_residuals(
        actual,
        predicted
    )

    expected = np.array([
        0.5,
        -0.5,
        0.0
    ])

    assert np.allclose(
        residuals,
        expected
    )


def test_residual_summary():

    actual = np.array([
        1.0,
        2.0,
        3.0
    ])

    predicted = np.array([
        0.5,
        2.5,
        3.0
    ])

    summary = summarize_residuals(
        actual,
        predicted
    )

    assert "Mean_Residual" in summary
    assert "Median_Residual" in summary
    assert "Std_Residual" in summary
    assert "Min_Residual" in summary
    assert "Max_Residual" in summary


def test_validation_table():

    actual = np.array([
        1.0,
        2.0,
        3.0
    ])

    predicted = np.array([
        1.1,
        1.9,
        3.2
    ])

    table = create_validation_table(
        "Test Model",
        actual,
        predicted
    )

    assert isinstance(
        table,
        pd.DataFrame
    )

    assert len(table) == 1

    assert table.loc[
        0,
        "Model"
    ] == "Test Model"

    assert "MSE" in table.columns
    assert "RMSE" in table.columns
    assert "MAE" in table.columns
    assert "R2" in table.columns