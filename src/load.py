import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Date"].dt.dayofweek == 1]
    return df.sort_values("Date")
