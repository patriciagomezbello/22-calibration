import json
import numpy as np
from load import load_data
from features import distance_features
from calibration import calibrate_p1_5, calibrate_p6_7
from collections import Counter
from llm_agent import llm_refine
from config import P1_5_REFINED_RANGE, P6_7_REFINED_RANGE
from neural_network import NeuralNetwork, DBN
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

df = load_data("data/22.csv")
df = distance_features(df)

# Define lines
lines_p1_5 = {
    1: list(range(1, 10)),
    2: list(range(10, 19)),
    3: list(range(19, 28)),
    4: list(range(28, 37)),
    5: list(range(37, 46)),
    6: list(range(46, 51)),
}

lines_p6_7 = {1: list(range(1, 5)), 2: list(range(5, 9)), 3: list(range(9, 13))}

# Find busy line for P1-P5
line_counts = {i: 0 for i in range(1, 7)}
for _, row in df.iterrows():
    for p in ["P1", "P2", "P3", "P4", "P5"]:
        pos = int(row[p])
        for line, rng in lines_p1_5.items():
            if pos in rng:
                line_counts[line] += 1
busy_line_p1_5 = max(line_counts, key=line_counts.get)

# Find busy line for P6-P7
line_counts_p6_7 = {i: 0 for i in range(1, 4)}
for _, row in df.iterrows():
    for p in ["P6", "P7"]:
        pos = int(row[p])
        for line, rng in lines_p6_7.items():
            if pos in rng:
                line_counts_p6_7[line] += 1
busy_line_p6_7 = max(line_counts_p6_7, key=line_counts_p6_7.get)

# Define boundaries based on busy lines to close the gap
p1_5_boundaries = lines_p1_5[busy_line_p1_5]
p1_5_min, p1_5_max = p1_5_boundaries[0], p1_5_boundaries[-1]
p6_7_boundaries = lines_p6_7[busy_line_p6_7]
p6_7_min, p6_7_max = p6_7_boundaries[0], p6_7_boundaries[-1]

# Prediction for P1-P5: predict next positions directly, then sort and ensure unique
positions_p1_5 = df[["P1", "P2", "P3", "P4", "P5"]].values
# Normalize to 0-1
positions_p1_5_norm = (positions_p1_5 - 1) / 49
# Create training data: X is current positions, y is next positions
split_idx = int(0.8 * len(positions_p1_5_norm))
X_p1_5 = positions_p1_5_norm[:-1][:split_idx]
y_p1_5 = positions_p1_5_norm[1:][:split_idx]
X_val_p1_5 = positions_p1_5_norm[:-1][split_idx:]
y_val_p1_5 = positions_p1_5_norm[1:][split_idx:]

# Train models
# NN with DBN pretrain
dbn_p1_5 = DBN(layers=[5, 10, 5], learning_rate=0.01, n_iter=20)
dbn_p1_5.train(X_p1_5)
nn_p1_5 = NeuralNetwork(
    input_size=5,
    hidden_size=5,
    output_size=5,
    learning_rate=0.01,
    dropout_rate=0.3,
    weight_decay=0.001,
)
nn_p1_5.train(X_p1_5, y_p1_5, epochs=500, regression=True)
val_pred_nn = nn_p1_5.predict(X_val_p1_5, regression=True)
mae_nn = np.mean(np.abs(val_pred_nn - y_val_p1_5))

# Random Forest
rf_p1_5 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0))
rf_p1_5.fit(X_p1_5, y_p1_5)
val_pred_rf = rf_p1_5.predict(X_val_p1_5)
mae_rf = np.mean(np.abs(val_pred_rf - y_val_p1_5))

# SVM
svm_p1_5 = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.1))
svm_p1_5.fit(X_p1_5, y_p1_5)
val_pred_svm = svm_p1_5.predict(X_val_p1_5)
mae_svm = np.mean(np.abs(val_pred_svm - y_val_p1_5))

# XGBoost
xgb_p1_5 = XGBRegressor(n_estimators=100, random_state=0)
xgb_p1_5.fit(X_p1_5, y_p1_5)
val_pred_xgb = xgb_p1_5.predict(X_val_p1_5)
mae_xgb = np.mean(np.abs(val_pred_xgb - y_val_p1_5))

print(
    f"P1-5 Validation MAEs: NN={mae_nn:.4f}, RF={mae_rf:.4f}, SVM={mae_svm:.4f}, XGB={mae_xgb:.4f}"
)

# Choose best model based on MAE
maes = {"nn": mae_nn, "rf": mae_rf, "svm": mae_svm, "xgb": mae_xgb}
best_model = min(maes, key=maes.get)
print(f"Best model for P1-5: {best_model}")

# Predict next positions
last_positions_norm = positions_p1_5_norm[-1].reshape(1, -1)
if best_model == "nn":
    pred_p1_5_norm = nn_p1_5.predict(last_positions_norm, regression=True).flatten()
elif best_model == "rf":
    pred_p1_5_norm = rf_p1_5.predict(last_positions_norm)[0]
elif best_model == "svm":
    pred_p1_5_norm = svm_p1_5.predict(last_positions_norm)[0]
else:
    pred_p1_5_norm = xgb_p1_5.predict(last_positions_norm)[0]

pred_p1_5 = (pred_p1_5_norm * 49 + 1).astype(int)
# Ensure within range and unique, avoid high values like 44
pred_p1_5 = np.clip(pred_p1_5, 1, 40)
pred_p1_5 = np.sort(np.unique(pred_p1_5))
while len(pred_p1_5) < 5:
    pred_p1_5 = np.append(pred_p1_5, np.random.randint(1, 51))
    pred_p1_5 = np.sort(np.unique(pred_p1_5))
pred_p1_5 = pred_p1_5[:5]

# Prediction for P6-P7: predict next positions directly, then sort and ensure unique
positions_p6_7 = df[["P6", "P7"]].values
# Normalize to 0-1
positions_p6_7_norm = (positions_p6_7 - 1) / 11
# Create training data: X is current positions, y is next positions
split_idx = int(0.8 * len(positions_p6_7_norm))
X_p6_7 = positions_p6_7_norm[:-1][:split_idx]
y_p6_7 = positions_p6_7_norm[1:][:split_idx]
X_val_p6_7 = positions_p6_7_norm[:-1][split_idx:]
y_val_p6_7 = positions_p6_7_norm[1:][split_idx:]

# Train models
# NN with DBN pretrain
dbn_p6_7 = DBN(layers=[2, 5, 2], learning_rate=0.01, n_iter=20)
dbn_p6_7.train(X_p6_7)
nn_p6_7 = NeuralNetwork(
    input_size=2,
    hidden_size=3,
    output_size=2,
    learning_rate=0.01,
    dropout_rate=0.3,
    weight_decay=0.001,
)
nn_p6_7.train(X_p6_7, y_p6_7, epochs=500, regression=True)
val_pred_nn_p6_7 = nn_p6_7.predict(X_val_p6_7, regression=True)
mae_nn_p6_7 = np.mean(np.abs(val_pred_nn_p6_7 - y_val_p6_7))

# Random Forest
rf_p6_7 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0))
rf_p6_7.fit(X_p6_7, y_p6_7)
val_pred_rf_p6_7 = rf_p6_7.predict(X_val_p6_7)
mae_rf_p6_7 = np.mean(np.abs(val_pred_rf_p6_7 - y_val_p6_7))

# SVM
svm_p6_7 = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.1))
svm_p6_7.fit(X_p6_7, y_p6_7)
val_pred_svm_p6_7 = svm_p6_7.predict(X_val_p6_7)
mae_svm_p6_7 = np.mean(np.abs(val_pred_svm_p6_7 - y_val_p6_7))

# XGBoost
xgb_p6_7 = XGBRegressor(n_estimators=100, random_state=0)
xgb_p6_7.fit(X_p6_7, y_p6_7)
val_pred_xgb_p6_7 = xgb_p6_7.predict(X_val_p6_7)
mae_xgb_p6_7 = np.mean(np.abs(val_pred_xgb_p6_7 - y_val_p6_7))

print(
    f"P6-7 Validation MAEs: NN={mae_nn_p6_7:.4f}, RF={mae_rf_p6_7:.4f}, SVM={mae_svm_p6_7:.4f}, XGB={mae_xgb_p6_7:.4f}"
)

# Choose best model based on MAE
maes_p6_7 = {
    "nn": mae_nn_p6_7,
    "rf": mae_rf_p6_7,
    "svm": mae_svm_p6_7,
    "xgb": mae_xgb_p6_7,
}
best_model_p6_7 = min(maes_p6_7, key=maes_p6_7.get)
print(f"Best model for P6-7: {best_model_p6_7}")

# Predict next positions
last_positions_p6_7_norm = positions_p6_7_norm[-1].reshape(1, -1)
if best_model_p6_7 == "nn":
    pred_p6_7_norm = nn_p6_7.predict(
        last_positions_p6_7_norm, regression=True
    ).flatten()
elif best_model_p6_7 == "rf":
    pred_p6_7_norm = rf_p6_7.predict(last_positions_p6_7_norm)[0]
elif best_model_p6_7 == "svm":
    pred_p6_7_norm = svm_p6_7.predict(last_positions_p6_7_norm)[0]
else:
    pred_p6_7_norm = xgb_p6_7.predict(last_positions_p6_7_norm)[0]

pred_p6_7 = (pred_p6_7_norm * 11 + 1).astype(int)
# Ensure within range and unique
pred_p6_7 = np.clip(pred_p6_7, 1, 12)
pred_p6_7 = np.sort(np.unique(pred_p6_7))
while len(pred_p6_7) < 2:
    pred_p6_7 = np.append(pred_p6_7, np.random.randint(1, 13))
    pred_p6_7 = np.sort(np.unique(pred_p6_7))
pred_p6_7 = pred_p6_7[:2]

# Most likely positions
most_likely_p1_5 = pred_p1_5.tolist()
most_likely_p6_7 = pred_p6_7.tolist()

# Least likely: least common in respective ranges
ranges_p = {
    "P1": range(1, 10),
    "P2": range(10, 19),
    "P3": range(19, 28),
    "P4": range(28, 37),
    "P5": range(37, 46),
    "P6": range(1, 5),
    "P7": range(5, 9),
}

least_likely = {}
for p, rng in ranges_p.items():
    counts = df[p].value_counts()
    candidates = counts[counts.index.isin(rng)]
    if not candidates.empty:
        least_likely[p] = int(candidates.idxmin())
    else:
        least_likely[p] = min(rng)


# Calculate distances
def calc_distances(positions):
    return [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]


dist_p1_5 = calc_distances(most_likely_p1_5)
dist_p6_7 = calc_distances(most_likely_p6_7)

least_likely_p1_5 = [
    least_likely["P1"],
    least_likely["P2"],
    least_likely["P3"],
    least_likely["P4"],
    least_likely["P5"],
]
least_likely_p6_7 = [least_likely["P6"], least_likely["P7"]]
dist_least_p1_5 = calc_distances(least_likely_p1_5)
dist_least_p6_7 = calc_distances(least_likely_p6_7)

result = {
    "most_likely": {
        "positions": {
            "P1": most_likely_p1_5[0],
            "P2": most_likely_p1_5[1],
            "P3": most_likely_p1_5[2],
            "P4": most_likely_p1_5[3],
            "P5": most_likely_p1_5[4],
            "P6": most_likely_p6_7[0],
            "P7": most_likely_p6_7[1],
        },
        "distances": {"P1_P5": dist_p1_5, "P6_P7": dist_p6_7},
    },
    "least_likely": {
        "positions": least_likely,
        "distances": {"P1_P5": dist_least_p1_5, "P6_P7": dist_least_p6_7},
    },
}

# Refine using deep abstract mathematics (LLM) - commented out due to API key
# result = llm_refine(result)

print(json.dumps(result))
