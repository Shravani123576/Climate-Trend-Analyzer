import numpy as np
import pandas as pd

from src.forecasting import (
    validate_input_data,
    split_train_test,
    fit_linear_baseline,
    fit_arima,
    forecast_arima,
    calculate_forecast_metrics,
    forecast_future,
)


def create_sample_data():

    years = np.arange(
        2000,
        2030
    )

    temperature = (
        0.01 * (years - 2000)
        + np.sin(
            np.arange(len(years)) * 0.5
        ) * 0.02
    )

    return pd.DataFrame({
        "Year": years,
        "Temperature_Anomaly": temperature
    })


def test_validate_input_data():

    df = create_sample_data()

    assert validate_input_data(df) is True


def test_split_train_test():

    df = create_sample_data()

    train, test = split_train_test(
        df,
        test_years=5
    )

    assert len(train) == 25

    assert len(test) == 5

    assert train["Year"].max() < test["Year"].min()


def test_linear_baseline():

    df = create_sample_data()

    train, test = split_train_test(
        df,
        test_years=5
    )

    model, predictions = fit_linear_baseline(
        train,
        test
    )

    assert len(predictions) == len(test)

    assert hasattr(
        model,
        "coef_"
    )


def test_arima_fit():

    df = create_sample_data()

    train, _ = split_train_test(
        df,
        test_years=5
    )

    model = fit_arima(
        train,
        order=(1, 1, 1)
    )

    assert model is not None


def test_arima_forecast():

    df = create_sample_data()

    train, _ = split_train_test(
        df,
        test_years=5
    )

    model = fit_arima(
        train,
        order=(1, 1, 1)
    )

    predictions = forecast_arima(
        model,
        steps=5
    )

    assert len(predictions) == 5

    assert np.isfinite(
        predictions
    ).all()


def test_forecast_metrics():

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

    metrics = calculate_forecast_metrics(
        actual,
        predicted
    )

    assert "MSE" in metrics

    assert "RMSE" in metrics

    assert "MAE" in metrics

    assert metrics["MSE"] >= 0

    assert metrics["RMSE"] >= 0

    assert metrics["MAE"] >= 0


def test_future_forecast():

    df = create_sample_data()

    model, forecast_df = forecast_future(
        df,
        years=5,
        order=(1, 1, 1)
    )

    assert model is not None

    assert len(forecast_df) == 5

    assert list(
        forecast_df.columns
    ) == [
        "Year",
        "Forecast_Temperature_Anomaly"
    ]


def test_future_forecast_years():

    df = create_sample_data()

    _, forecast_df = forecast_future(
        df,
        years=5
    )

    assert forecast_df[
        "Year"
    ].tolist() == [
        2030,
        2031,
        2032,
        2033,
        2034
    ]