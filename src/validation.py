from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def validate_arrays(y_true, y_pred):
    """
    Validate actual and predicted values before calculating metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        raise ValueError("Actual values are empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted values must have the same length."
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            "Actual values contain NaN or infinite values."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "Predicted values contain NaN or infinite values."
        )

    return y_true, y_pred


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate MSE, RMSE, MAE and R².
    """

    y_true, y_pred = validate_arrays(
        y_true,
        y_pred
    )

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


def calculate_residuals(y_true, y_pred):
    """
    Calculate residuals as:

    Actual - Predicted
    """

    y_true, y_pred = validate_arrays(
        y_true,
        y_pred
    )

    return y_true - y_pred


def summarize_residuals(
    y_true,
    y_pred
):
    """
    Generate statistical summary of residuals.
    """

    residuals = calculate_residuals(
        y_true,
        y_pred
    )

    return {
        "Mean_Residual": float(
            np.mean(residuals)
        ),
        "Median_Residual": float(
            np.median(residuals)
        ),
        "Std_Residual": float(
            np.std(residuals)
        ),
        "Min_Residual": float(
            np.min(residuals)
        ),
        "Max_Residual": float(
            np.max(residuals)
        )
    }


def create_validation_table(
    model_name,
    y_true,
    y_pred
):
    """
    Create a single-row validation table.
    """

    metrics = calculate_regression_metrics(
        y_true,
        y_pred
    )

    residual_summary = summarize_residuals(
        y_true,
        y_pred
    )

    result = {
        "Model": model_name,
        **metrics,
        **residual_summary
    }

    return pd.DataFrame([
        result
    ])


def create_residual_plot(
    y_true,
    y_pred,
    output_path="outputs/graphs/validation_residuals.png"
):
    """
    Create and save a residual plot.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    residuals = calculate_residuals(
        y_true,
        y_pred
    )

    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.title(
        "Residual Validation Plot"
    )

    plt.xlabel(
        "Predicted Temperature Anomaly (°C)"
    )

    plt.ylabel(
        "Residual"
    )

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_actual_vs_predicted_plot(
    y_true,
    y_pred,
    output_path="outputs/graphs/actual_vs_predicted.png"
):
    """
    Create and save an actual-vs-predicted plot.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    y_true, y_pred = validate_arrays(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.7
    )

    minimum = min(
        y_true.min(),
        y_pred.min()
    )

    maximum = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.title(
        "Actual vs Predicted Temperature Anomaly"
    )

    plt.xlabel(
        "Actual"
    )

    plt.ylabel(
        "Predicted"
    )

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def save_validation_results(
    validation_df,
    output_path="outputs/tables/validation_results.csv"
):
    """
    Save validation metrics to CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_df.to_csv(
        output_path,
        index=False
    )

    return output_path


def validate_model(
    model_name,
    y_true,
    y_pred,
    save_outputs=True
):
    """
    Complete validation workflow for one model.
    """

    validation_df = create_validation_table(
        model_name,
        y_true,
        y_pred
    )

    residual_summary = summarize_residuals(
        y_true,
        y_pred
    )

    if save_outputs:

        create_residual_plot(
            y_true,
            y_pred
        )

        create_actual_vs_predicted_plot(
            y_true,
            y_pred
        )

        save_validation_results(
            validation_df
        )

    return {
        "metrics": calculate_regression_metrics(
            y_true,
            y_pred
        ),
        "residuals": residual_summary,
        "table": validation_df
    }