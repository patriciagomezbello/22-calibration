import pandas as pd


def load_data(path: str):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])  # Drop rows with invalid dates
    df = df[
        (df["Date"].dt.dayofweek == 1) | (df["Date"].dt.dayofweek == 4)
    ]  # Tuesday and Friday
    return df.sort_values("Date").reset_index(drop=True)
