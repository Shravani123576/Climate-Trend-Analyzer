import pandas as pd
import numpy as np

from src.trend_analysis import (
    validate_input_data,
    prepare_features,
    fit_linear_regression,
    fit_polynomial_regression,
    calculate_metrics,
    calculate_residuals,
    analyze_linear_trend,
    analyze_polynomial_trend,
)


def create_sample_data():

    return pd.DataFrame({
        "Year": [
            2000,
            2001,
            2002,
            2003,
            2004,
            2005
        ],
        "Temperature_Anomaly": [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35
        ]
    })


def test_validate_input_data():

    df = create_sample_data()

    assert validate_input_data(df) is True


def test_prepare_features():

    df = create_sample_data()

    X, y = prepare_features(df)

    assert list(X.columns) == ["Year"]

    assert len(X) == 6

    assert len(y) == 6


def test_linear_regression():

    df = create_sample_data()

    model, predictions = fit_linear_regression(
        df
    )

    assert len(predictions) == len(df)

    assert hasattr(
        model,
        "coef_"
    )


def test_polynomial_regression():

    df = create_sample_data()

    model, predictions = fit_polynomial_regression(
        df,
        degree=2
    )

    assert len(predictions) == len(df)


def test_calculate_metrics():

    y_true = np.array([
        1.0,
        2.0,
        3.0
    ])

    y_pred = np.array([
        1.0,
        2.0,
        3.0
    ])

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    assert metrics["MSE"] == 0

    assert metrics["RMSE"] == 0

    assert metrics["MAE"] == 0

    assert metrics["R2"] == 1


def test_residuals():

    y_true = np.array([
        1.0,
        2.0,
        3.0
    ])

    y_pred = np.array([
        0.5,
        2.5,
        3.0
    ])

    residuals = calculate_residuals(
        y_true,
        y_pred
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


def test_linear_analysis():

    df = create_sample_data()

    result = analyze_linear_trend(
        df
    )

    assert "model" in result

    assert "predictions" in result

    assert "metrics" in result

    assert len(
        result["predictions"]
    ) == len(df)


def test_polynomial_analysis():

    df = create_sample_data()

    result = analyze_polynomial_trend(
        df,
        degree=2
    )

    assert "model" in result

    assert "predictions" in result

    assert "metrics" in result