# Forecasting Report

## Objective

Estimate future global temperature anomalies using historical observations.

## Models Evaluated

- Linear Regression
- ARIMA

## Evaluation

The models were compared using MAE, RMSE, and R² on a held-out test set.

## Findings

- The baseline Linear Regression model captured the long-term trend.
- ARIMA modeled the temporal structure of the data and achieved improved forecasting performance (if supported by your metrics).
- The selected model was retrained on the full historical dataset to generate a 10-year forecast.

## Conclusion

This forecasting framework demonstrates how historical climate data can be used to estimate future temperature anomalies while objectively comparing competing models.