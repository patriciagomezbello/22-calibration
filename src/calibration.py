def calibrate_p1_5(df):
    dists = df[[f"D{i}" for i in range(1, 5)]].mean().round().astype(int)
    base = df.iloc[-1][["P1","P2","P3","P4","P5"]].values

    pred = [int(base[0])]
    for d in dists:
        pred.append(pred[-1] + int(d))

    pred = sorted(set(max(1, min(50, x)) for x in pred))
    return pred[:5]

def calibrate_p6_7(df):
    d = int(df["D67"].mean().round())
    base = int(df.iloc[-1]["P6"])
    p7 = max(1, min(12, base + d))
    return [base, p7]
