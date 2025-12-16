def distance_features(df):
    for i in range(1, 5):
        df[f"D{i}"] = df[f"P{i+1}"] - df[f"P{i}"]
    df["D67"] = df["P7"] - df["P6"]
    return df
