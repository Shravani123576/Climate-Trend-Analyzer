from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from statsmodels.tsa.arima.model import ARIMA


def validate_input_data(df):
    """
    Validate the dataframe required for forecasting.
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


def load_forecasting_data(
    file_path="data/processed/climate_trend_clean.csv"
):
    """
    Load the cleaned climate dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    validate_input_data(df)

    df = df.sort_values(
        "Year"
    ).reset_index(drop=True)

    return df


def split_train_test(
    df,
    test_years=10
):
    """
    Split the time series chronologically.

    The final `test_years` observations are used
    as the test set.
    """

    if test_years <= 0:
        raise ValueError(
            "test_years must be greater than zero."
        )

    if test_years >= len(df):
        raise ValueError(
            "test_years must be smaller than dataset length."
        )

    train = df.iloc[:-test_years].copy()

    test = df.iloc[-test_years:].copy()

    return train, test


def fit_linear_baseline(
    train_df,
    test_df
):
    """
    Fit a simple linear-regression forecasting baseline.
    """

    X_train = train_df[
        ["Year"]
    ]

    y_train = train_df[
        "Temperature_Anomaly"
    ]

    X_test = test_df[
        ["Year"]
    ]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return model, predictions


def fit_arima(
    train_df,
    order=(1, 1, 1)
):
    """
    Fit an ARIMA model to the training temperature series.
    """

    series = train_df[
        "Temperature_Anomaly"
    ].astype(float)

    model = ARIMA(
        series,
        order=order
    )

    fitted_model = model.fit()

    return fitted_model


def forecast_arima(
    fitted_model,
    steps
):
    """
    Forecast future temperature anomalies.
    """

    if steps <= 0:
        raise ValueError(
            "steps must be greater than zero."
        )

    forecast = fitted_model.forecast(
        steps=steps
    )

    return np.asarray(
        forecast
    )


def calculate_forecast_metrics(
    actual,
    predicted
):
    """
    Calculate forecasting error metrics.
    """

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }


def evaluate_linear_baseline(
    train_df,
    test_df
):
    """
    Evaluate the linear-regression baseline
    on the held-out test set.
    """

    model, predictions = fit_linear_baseline(
        train_df,
        test_df
    )

    metrics = calculate_forecast_metrics(
        test_df["Temperature_Anomaly"],
        predictions
    )

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }


def evaluate_arima(
    train_df,
    test_df,
    order=(1, 1, 1)
):
    """
    Fit ARIMA on training data and evaluate
    predictions against the held-out test set.
    """

    model = fit_arima(
        train_df,
        order=order
    )

    predictions = forecast_arima(
        model,
        len(test_df)
    )

    metrics = calculate_forecast_metrics(
        test_df["Temperature_Anomaly"],
        predictions
    )

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }


def forecast_future(
    df,
    years=15,
    order=(1, 1, 1)
):
    """
    Train ARIMA on the complete historical dataset
    and forecast the requested number of future years.
    """

    if years <= 0:
        raise ValueError(
            "years must be greater than zero."
        )

    model = fit_arima(
        df,
        order=order
    )

    forecast_values = forecast_arima(
        model,
        years
    )

    last_year = int(
        df["Year"].max()
    )

    future_years = np.arange(
        last_year + 1,
        last_year + years + 1
    )

    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Forecast_Temperature_Anomaly": forecast_values
    })

    return model, forecast_df


def create_forecast_plot(
    historical_df,
    forecast_df,
    output_path="outputs/graphs/forecast.png"
):
    """
    Create and save historical + forecast visualization.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        historical_df["Year"],
        historical_df["Temperature_Anomaly"],
        label="Historical"
    )

    plt.plot(
        forecast_df["Year"],
        forecast_df["Forecast_Temperature_Anomaly"],
        linestyle="--",
        linewidth=2,
        label="ARIMA Forecast"
    )

    plt.title(
        "Global Temperature Anomaly Forecast"
    )

    plt.xlabel(
        "Year"
    )

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


def save_forecast_results(
    forecast_df,
    output_path="outputs/tables/forecast_results.csv"
):
    """
    Save forecasted values to CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast_df.to_csv(
        output_path,
        index=False
    )

    return output_path


def save_forecast_model_metrics(
    linear_metrics,
    arima_metrics,
    output_path="outputs/tables/forecast_model_comparison.csv"
):
    """
    Save held-out test metrics for baseline vs ARIMA.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison = pd.DataFrame({
        "Linear Baseline": linear_metrics,
        "ARIMA": arima_metrics
    })

    comparison.to_csv(
        output_path
    )

    return output_path


def run_forecasting(
    input_path="data/processed/climate_trend_clean.csv",
    test_years=10,
    forecast_years=15,
    arima_order=(1, 1, 1)
):
    """
    Run the complete forecasting workflow.
    """

    df = load_forecasting_data(
        input_path
    )

    train_df, test_df = split_train_test(
        df,
        test_years=test_years
    )

    linear_results = evaluate_linear_baseline(
        train_df,
        test_df
    )

    arima_results = evaluate_arima(
        train_df,
        test_df,
        order=arima_order
    )

    _, future_forecast = forecast_future(
        df,
        years=forecast_years,
        order=arima_order
    )

    create_forecast_plot(
        df,
        future_forecast
    )

    save_forecast_results(
        future_forecast
    )

    save_forecast_model_metrics(
        linear_results["metrics"],
        arima_results["metrics"]
    )

    print("\n===== FORECASTING =====")

    print("\nLinear Baseline:")

    for key, value in linear_results["metrics"].items():
        print(
            f"{key}: {value:.6f}"
        )

    print("\nARIMA:")

    for key, value in arima_results["metrics"].items():
        print(
            f"{key}: {value:.6f}"
        )

    print(
        f"\nFuture forecast years: "
        f"{forecast_years}"
    )

    print(
        "\nForecasting completed successfully."
    )

    return {
        "linear": linear_results,
        "arima": arima_results,
        "future_forecast": future_forecast
    }