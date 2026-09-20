# 🌍 Climate Trend Analyzer

An end-to-end data science project for analyzing historical global temperature anomalies, identifying unusual climate observations, forecasting future temperature anomalies, and presenting results through an interactive Streamlit dashboard.

## 📌 Project Overview

The Climate Trend Analyzer uses historical NASA GISS temperature-anomaly data to answer practical analytical questions:

- How has global temperature anomaly changed over time?
- What are the main statistical characteristics of the historical data?
- Which years have unusually high or low temperature anomalies?
- How well can statistical models represent the historical trend?
- Can a time-series model generate a future forecast?
- How can the results be presented through a reusable automated pipeline and dashboard?

## 🎯 Problem Statement

Raw climate datasets contain historical observations that need to be cleaned, explored, analyzed, modeled, and validated before they can support meaningful insights.

This project builds a reproducible workflow that transforms historical climate data into:

- statistical summaries
- trend analysis
- anomaly detection
- time-series forecasts
- model validation results
- visualizations
- an interactive dashboard

## 📊 Dataset

Source: NASA GISS GISTEMP v4

The project uses the global land-ocean temperature anomaly series.

Main variables used in the analysis:

- `Year`
- `Temperature_Anomaly`

The annual anomaly is derived from NASA's `J-D` annual value.

The raw dataset contains an initial descriptive line before the CSV header, which is handled during data loading.

## 🧰 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Statsmodels
- SciPy
- Streamlit
- PyTest

## 🏗️ Project Architecture

```text
NASA GISS Dataset
       ↓
Data Loading
       ↓
Preprocessing
       ↓
EDA
       ↓
Trend Analysis
       ↓
Anomaly Detection
       ↓
Forecasting
       ↓
Validation
       ↓
Saved Outputs
       ↓
Streamlit Dashboard