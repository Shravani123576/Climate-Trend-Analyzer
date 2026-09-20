from scipy.stats import linregress
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def validate_input_data(df):
    """
    Validate that the dataframe contains the columns
    required for trend analysis.
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


def prepare_features(df):
    """
    Prepare Year as the independent variable
    and Temperature_Anomaly as the target.
    """

    validate_input_data(df)

    X = df[["Year"]].copy()
    y = df["Temperature_Anomaly"].copy()

    return X, y


def fit_linear_regression(df):
    """
    Fit a linear regression model to the climate trend.
    """

    X, y = prepare_features(df)

    model = LinearRegression()

    model.fit(X, y)

    predictions = model.predict(X)

    return model, predictions


def fit_polynomial_regression(
    df,
    degree=2
):
    """
    Fit polynomial regression to capture
    non-linear climate trends.
    """

    X, y = prepare_features(df)

    model = Pipeline([
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),
        (
            "linear_regression",
            LinearRegression()
        )
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    return model, predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics.
    """

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def analyze_linear_trend(df):
    """
    Complete linear trend analysis.

    Returns:
        model
        predictions
        metrics
    """

    model, predictions = fit_linear_regression(
        df
    )

    metrics = calculate_metrics(
        df["Temperature_Anomaly"],
        predictions
    )

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }


def analyze_polynomial_trend(
    df,
    degree=2
):
    """
    Complete polynomial trend analysis.
    """

    model, predictions = fit_polynomial_regression(
        df,
        degree=degree
    )

    metrics = calculate_metrics(
        df["Temperature_Anomaly"],
        predictions
    )

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }


def calculate_residuals(
    y_true,
    y_pred
):
    """
    Calculate residuals:

    Actual - Predicted
    """

    return np.asarray(y_true) - np.asarray(y_pred)


def create_linear_trend_plot(
    df,
    predictions,
    output_path="outputs/graphs/linear_trend.png"
):
    """
    Save actual values and fitted linear trend.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(15, 6))

    plt.scatter(
        df["Year"],
        df["Temperature_Anomaly"],
        alpha=0.6,
        label="Actual"
    )

    plt.plot(
        df["Year"],
        predictions,
        linewidth=2,
        label="Linear Trend"
    )

    plt.title(
        "Global Temperature Anomaly - Linear Trend"
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


def create_polynomial_trend_plot(
    df,
    predictions,
    degree=2,
    output_path="outputs/graphs/polynomial_trend.png"
):
    """
    Save actual values and fitted polynomial trend.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_data = pd.DataFrame({
        "Year": df["Year"].values,
        "Prediction": predictions
    }).sort_values("Year")

    plt.figure(figsize=(15, 6))

    plt.scatter(
        df["Year"],
        df["Temperature_Anomaly"],
        alpha=0.6,
        label="Actual"
    )

    plt.plot(
        plot_data["Year"],
        plot_data["Prediction"],
        linewidth=2,
        label=f"Polynomial Trend (degree={degree})"
    )

    plt.title(
        "Global Temperature Anomaly - Polynomial Trend"
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


def create_residual_plot(
    df,
    predictions,
    output_path="outputs/graphs/trend_residuals.png"
):
    """
    Create and save a residual plot for linear regression.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    residuals = calculate_residuals(
        df["Temperature_Anomaly"],
        predictions
    )

    plt.figure(figsize=(15, 6))

    plt.scatter(
        df["Year"],
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.title(
        "Linear Regression Residuals"
    )

    plt.xlabel("Year")

    plt.ylabel("Residual")

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def save_trend_results(
    linear_results,
    polynomial_results,
    output_path="outputs/tables/trend_model_comparison.csv"
):
    """
    Save the model comparison metrics as CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison = pd.DataFrame({
        "Linear Regression": linear_results["metrics"],
        "Polynomial Regression": polynomial_results["metrics"]
    })

    comparison.to_csv(
        output_path
    )

    return output_path


def run_trend_analysis(
    input_path="data/processed/climate_trend_clean.csv",
    polynomial_degree=2
):
    """
    Run the complete trend-analysis workflow.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    validate_input_data(df)

    linear_results = analyze_linear_trend(
        df
    )

    polynomial_results = analyze_polynomial_trend(
        df,
        degree=polynomial_degree
    )

    create_linear_trend_plot(
        df,
        linear_results["predictions"]
    )

    create_polynomial_trend_plot(
        df,
        polynomial_results["predictions"],
        degree=polynomial_degree
    )

    create_residual_plot(
        df,
        linear_results["predictions"]
    )

    save_trend_results(
        linear_results,
        polynomial_results
    )

    print("\n===== TREND ANALYSIS =====")

    print("\nLinear Regression:")

    for key, value in linear_results["metrics"].items():
        print(f"{key}: {value:.6f}")

    print("\nPolynomial Regression:")

    for key, value in polynomial_results["metrics"].items():
        print(f"{key}: {value:.6f}")

    print(
        "\nTrend analysis completed successfully."
    )

    return {
        "linear": linear_results,
        "polynomial": polynomial_results
    }

def calculate_linear_trend(df):
    """
    Calculate linear regression statistics.

    Automatically removes missing values before analysis.
    """

    trend_df = df.dropna(
        subset=["Year", "Temperature_Anomaly"]
    )

    result = linregress(
        trend_df["Year"],
        trend_df["Temperature_Anomaly"]
    )

    return result
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_regression(actual, predicted):
    """
    Evaluate regression model performance.
    """

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = mse ** 0.5

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }