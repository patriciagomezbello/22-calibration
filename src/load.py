import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[(df["Date"].dt.dayofweek == 1) | (df["Date"].dt.dayofweek == 4)]  # Tuesday and Friday
    return df.sort_values("Date")
