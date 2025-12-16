from load import load_data
from features import distance_features
from calibration import calibrate_p1_5, calibrate_p6_7

df = load_data("data/22.csv")
df = distance_features(df)

result = {
    "P1_P5": calibrate_p1_5(df),
    "P6_P7": calibrate_p6_7(df)
}

print("FINAL PREDICTION:")
print(result)
