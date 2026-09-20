import pandas as pd


def calculate_yearly_change(df):
    """
    Calculate year-over-year change in temperature anomaly.
    """

    data = df.copy()

    data["Yearly_Change"] = (
        data["Temperature_Anomaly"].diff()
    )

    return data