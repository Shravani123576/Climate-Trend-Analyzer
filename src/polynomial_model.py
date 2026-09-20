import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
def train_polynomial_model(
    df,
    degree=2
):
    """
    Train Polynomial Regression Model.
    """

    X = df[["Year"]]

    y = df["Temperature_Anomaly"]

    polynomial = PolynomialFeatures(
        degree=degree
    )

    X_poly = polynomial.fit_transform(X)

    model = LinearRegression()

    model.fit(
        X_poly,
        y
    )

    predictions = model.predict(X_poly)

    return (
        model,
        polynomial,
        predictions
    )
def polynomial_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }