from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "climate_trend_clean.csv"
)

ANOMALY_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "anomaly_detection_results.csv"
)

ANOMALY_SUMMARY_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "anomaly_summary.csv"
)

FORECAST_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "forecast_results.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "forecast_model_comparison.csv"
)

TREND_MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "trend_model_comparison.csv"
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🌍 Climate Trend Analyzer"
)

st.caption(
    "Historical Global Temperature Anomaly Analysis"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    return df


@st.cache_data
def load_anomalies():

    if not ANOMALY_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        ANOMALY_PATH
    )


@st.cache_data
def load_anomaly_summary():

    if not ANOMALY_SUMMARY_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        ANOMALY_SUMMARY_PATH
    )


@st.cache_data
def load_forecast():

    if not FORECAST_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        FORECAST_PATH
    )


@st.cache_data
def load_model_comparison():

    if not MODEL_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        MODEL_PATH
    )


@st.cache_data
def load_trend_models():

    if not TREND_MODEL_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        TREND_MODEL_PATH
    )


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if not DATA_PATH.exists():

    st.error(
        "Processed dataset not found. "
        "Run `python main.py` first."
    )

    st.stop()


df = load_data()

anomaly_df = load_anomalies()

anomaly_summary_df = load_anomaly_summary()

forecast_df = load_forecast()

model_comparison_df = load_model_comparison()

trend_model_df = load_trend_models()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "Dashboard Controls"
)

show_anomalies = st.sidebar.checkbox(
    "Show anomaly data",
    value=True
)

show_forecast = st.sidebar.checkbox(
    "Show forecast",
    value=True
)


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

st.subheader(
    "Dataset Overview"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Years Analyzed",
        len(df)
    )

with col2:

    st.metric(
        "Mean Anomaly",
        f"{df['Temperature_Anomaly'].mean():.3f} °C"
    )

with col3:

    st.metric(
        "Minimum Anomaly",
        f"{df['Temperature_Anomaly'].min():.3f} °C"
    )

with col4:

    st.metric(
        "Maximum Anomaly",
        f"{df['Temperature_Anomaly'].max():.3f} °C"
    )


# --------------------------------------------------
# TEMPERATURE TREND
# --------------------------------------------------

st.subheader(
    "Global Temperature Anomaly Trend"
)

trend_chart = (
    df[
        [
            "Year",
            "Temperature_Anomaly"
        ]
    ]
    .set_index("Year")
)

st.line_chart(
    trend_chart
)


# --------------------------------------------------
# TEMPERATURE DISTRIBUTION
# --------------------------------------------------

st.subheader(
    "Temperature Anomaly Distribution"
)

distribution_chart = pd.DataFrame({
    "Temperature Anomaly": df[
        "Temperature_Anomaly"
    ].values
})

st.bar_chart(
    distribution_chart
)


# --------------------------------------------------
# ANOMALY SECTION
# --------------------------------------------------

if show_anomalies:

    st.subheader(
        "Anomaly Detection"
    )

    if not anomaly_summary_df.empty:

        summary = anomaly_summary_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Z-Score Anomalies",
                int(
                    summary[
                        "ZScore_Anomalies"
                    ]
                )
            )

        with col2:

            st.metric(
                "IQR Anomalies",
                int(
                    summary[
                        "IQR_Anomalies"
                    ]
                )
            )

        with col3:

            st.metric(
                "Detected by Both",
                int(
                    summary[
                        "Both_Methods"
                    ]
                )
            )

    if not anomaly_df.empty:

        anomaly_rows = anomaly_df[
            anomaly_df[
                "Detection_Category"
            ] != "None"
        ]

        st.dataframe(
            anomaly_rows,
            use_container_width=True
        )


# --------------------------------------------------
# FORECAST SECTION
# --------------------------------------------------

if show_forecast:

    st.subheader(
        "15-Year Temperature Anomaly Forecast"
    )

    if not forecast_df.empty:

        forecast_chart = pd.DataFrame()

        historical = df[
            [
                "Year",
                "Temperature_Anomaly"
            ]
        ].copy()

        historical["Type"] = "Historical"

        forecast = forecast_df[
            [
                "Year",
                "Forecast_Temperature_Anomaly"
            ]
        ].copy()

        forecast = forecast.rename(
            columns={
                "Forecast_Temperature_Anomaly":
                "Temperature_Anomaly"
            }
        )

        forecast["Type"] = "Forecast"

        combined = pd.concat(
            [
                historical,
                forecast
            ],
            ignore_index=True
        )

        st.line_chart(
            combined.set_index(
                "Year"
            )[
                "Temperature_Anomaly"
            ]
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader(
    "Forecast Model Performance"
)

if not model_comparison_df.empty:

    st.dataframe(
        model_comparison_df,
        use_container_width=True
    )

if not trend_model_df.empty:

    st.subheader(
        "Trend Model Performance"
    )

    st.dataframe(
        trend_model_df,
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Climate Trend Analyzer | "
    "Built with Python, Pandas, Scikit-Learn, "
    "Statsmodels and Streamlit"
)