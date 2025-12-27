import json
import numpy as np
from load import load_data
from features import distance_features
from calibration import calibrate_p1_5, calibrate_p6_7
from collections import Counter
from llm_agent import llm_refine
from config import P1_5_REFINED_RANGE, P6_7_REFINED_RANGE
from neural_network import NeuralNetwork, DBN

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

# Neural Network prediction for P1-P5
positions_p1_5 = df[["P1", "P2", "P3", "P4", "P5"]].values
# Normalize to 0-1
positions_p1_5_norm = (positions_p1_5 - 1) / 49  # 1-50 to 0-1
# Create training data: X is positions at t, y is at t+1
X_p1_5 = positions_p1_5_norm[:-1]
y_p1_5 = positions_p1_5_norm[1:]
# Train NN for P1-P5
nn_p1_5 = NeuralNetwork(input_size=5, hidden_size=10, output_size=5, learning_rate=0.01)
nn_p1_5.train(X_p1_5, y_p1_5, epochs=200, regression=True)
# Predict next
last_p1_5_norm = positions_p1_5_norm[-1].reshape(1, -1)
pred_p1_5_norm = nn_p1_5.predict(last_p1_5_norm, regression=True)
pred_p1_5 = (pred_p1_5_norm * 49 + 1).astype(int).flatten()
# Ensure within range and unique
pred_p1_5 = np.clip(pred_p1_5, 1, 50)
pred_p1_5 = np.unique(pred_p1_5)  # Remove duplicates
while len(pred_p1_5) < 5:
    pred_p1_5 = np.append(pred_p1_5, np.random.randint(1, 51))
pred_p1_5 = np.sort(pred_p1_5[:5])

# Neural Network for P6-P7
positions_p6_7 = df[["P6", "P7"]].values
positions_p6_7_norm = (positions_p6_7 - 1) / 11  # 1-12 to 0-1
X_p6_7 = positions_p6_7_norm[:-1]
y_p6_7 = positions_p6_7_norm[1:]
nn_p6_7 = NeuralNetwork(input_size=2, hidden_size=5, output_size=2, learning_rate=0.01)
nn_p6_7.train(X_p6_7, y_p6_7, epochs=500, regression=True)
last_p6_7_norm = positions_p6_7_norm[-1].reshape(1, -1)
pred_p6_7_norm = nn_p6_7.predict(last_p6_7_norm, regression=True)
pred_p6_7 = (pred_p6_7_norm * 11 + 1).astype(int).flatten()
pred_p6_7 = np.clip(pred_p6_7, 1, 12)
pred_p6_7 = np.unique(pred_p6_7)
while len(pred_p6_7) < 2:
    pred_p6_7 = np.append(pred_p6_7, np.random.randint(1, 13))
pred_p6_7 = np.sort(pred_p6_7[:2])

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
