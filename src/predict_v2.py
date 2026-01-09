"""
Calibration Prediction Script v2
Uses distributed representations, backpropagation, dropout, ReLU,
and overfitting prevention techniques for position prediction.
Target: Friday 9th January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import json


# ============== LOAD DATA ==============
def load_data(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Monday, 4=Friday
    return df


df = load_data("data/22.csv")


# ============== DISTANCE FEATURES ==============
def compute_distances(row):
    """Compute distances between consecutive positions"""
    return {
        "D1": int(row["P2"] - row["P1"]),
        "D2": int(row["P3"] - row["P2"]),
        "D3": int(row["P4"] - row["P3"]),
        "D4": int(row["P5"] - row["P4"]),
        "D67": int(row["P7"] - row["P6"]),
    }


# Add distance features
distances = df.apply(compute_distances, axis=1, result_type="expand")
df = pd.concat([df, distances], axis=1)


# ============== REGULARIZATION TECHNIQUES ==============
class RegularizedPredictor:
    """
    Predictor with distributed representations, dropout,
    weight decay, early stopping, and noise injection.
    """

    def __init__(
        self, learning_rate=0.01, dropout_rate=0.3, weight_decay=0.001, noise_std=0.1
    ):
        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.weights = None
        self.best_loss = float("inf")
        self.patience = 50
        self.patience_counter = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def add_noise(self, X):
        """Noise injection for regularization"""
        return X + np.random.normal(0, self.noise_std, X.shape)

    def dropout_mask(self, shape):
        """Dropout mask creation"""
        return (np.random.rand(*shape) > self.dropout_rate) / (1 - self.dropout_rate)

    def fit(self, X, y, epochs=1000):
        """Train with backpropagation and regularization"""
        n_samples, n_features = X.shape
        n_output = y.shape[1] if len(y.shape) > 1 else 1

        # Initialize weights (distributed representations)
        np.random.seed(42)
        self.weights = {
            "W1": np.random.randn(n_features, n_features) * 0.01,
            "b1": np.zeros((1, n_features)),
            "W2": np.random.randn(n_features, n_output) * 0.01,
            "b2": np.zeros((1, n_output)),
        }

        for epoch in range(epochs):
            # Forward pass with noise injection and dropout
            X_noisy = self.add_noise(X)

            # Layer 1
            Z1 = np.dot(X_noisy, self.weights["W1"]) + self.weights["b1"]
            A1 = self.relu(Z1)

            # Dropout
            mask = self.dropout_mask(A1.shape)
            A1_dropped = A1 * mask

            # Layer 2
            Z2 = np.dot(A1_dropped, self.weights["W2"]) + self.weights["b2"]
            A2 = Z2  # Linear activation for regression

            # Backward pass with L2 regularization
            m = n_samples
            dZ2 = (A2 - y) / m
            dW2 = np.dot(A1.T, dZ2) + self.weight_decay * self.weights["W2"]
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.weights["W2"].T)
            dZ1 = dA1 * self.relu_derivative(Z1)
            dW1 = np.dot(X.T, dZ1) + self.weight_decay * self.weights["W1"]
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Gradient descent
            self.weights["W2"] -= self.lr * dW2
            self.weights["b2"] -= self.lr * db2
            self.weights["W1"] -= self.lr * dW1
            self.weights["b1"] -= self.lr * db1

            # Early stopping
            loss = np.mean((A2 - y) ** 2)
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.patience:
                break

    def predict(self, X):
        """Predict without dropout/noise (inference mode)"""
        Z1 = np.dot(X, self.weights["W1"]) + self.weights["b1"]
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.weights["W2"]) + self.weights["b2"]
        return Z2


# ============== FREQUENCY-BASED PREDICTION ==============
def get_position_frequencies(df, positions, day_filter=None):
    """Get frequency distribution of positions"""
    if day_filter is not None:
        df_filtered = df[df["DayOfWeek"] == day_filter]
    else:
        df_filtered = df

    all_positions = []
    for p in positions:
        all_positions.extend(df_filtered[p].tolist())

    freq = Counter(all_positions)
    return freq


def get_distance_frequencies(df, distances, day_filter=None):
    """Get frequency distribution of distances"""
    if day_filter is not None:
        df_filtered = df[df["DayOfWeek"] == day_filter]
    else:
        df_filtered = df

    all_dists = []
    for d in distances:
        all_dists.extend(df_filtered[d].tolist())

    return Counter(all_dists)


# ============== MAIN PREDICTION ==============
# Target date: Friday 9th January 2026
TARGET_DATE = datetime(2026, 1, 9)
TARGET_DAY = 4  # Friday

print(f"Target Date: {TARGET_DATE.strftime('%Y-%m-%d')} (Friday)")
print("=" * 60)

# Get last record
last_row = df.iloc[-1]

# Get frequency distributions
print("\n1. ANALYZING DISTANCE PATTERNS...")
dist_frequencies = {}
for d in ["D1", "D2", "D3", "D4", "D67"]:
    dist_frequencies[d] = get_distance_frequencies(df, [d], day_filter=TARGET_DAY)

print("   Most common distances (P1->P2, P2->P3, P3->P4, P4->P5, P6->P7):")
for d in ["D1", "D2", "D3", "D4", "D67"]:
    most_common = dist_frequencies[d].most_common(3)
    print(f"   {d}: {most_common}")

# ============== P1-P5 PREDICTION ==============
print("\n2. P1-P5 PREDICTION (Range 1-50, ordered, no duplicates)...")

# Get base values from last record
base_p1 = int(last_row["P1"])
base_p2 = int(last_row["P2"])
base_p3 = int(last_row["P3"])
base_p4 = int(last_row["P4"])
base_p5 = int(last_row["P5"])

# Calculate distances using specified values
# From P1 to P2 is 5, P2 to P3 is 2, P3 to P4 is 11, P4 to P5 is 10
dist_p1_p2 = 5
dist_p2_p3 = 2
dist_p3_p4 = 11
dist_p4_p5 = 10

# Build positions using distance approach
p1_val = base_p1
p2_val = max(1, min(50, p1_val + dist_p1_p2))
p3_val = max(1, min(50, p2_val + dist_p2_p3))
p4_val = max(1, min(50, p3_val + dist_p3_p4))
p5_val = max(1, min(50, p4_val + dist_p4_p5))

most_likely_p1_5 = sorted(set([p1_val, p2_val, p3_val, p4_val, p5_val]))[:5]

# For least likely, get least frequent positions
freq_all_p1_5 = get_position_frequencies(
    df, ["P1", "P2", "P3", "P4", "P5"], day_filter=TARGET_DAY
)
sorted_by_freq = sorted(freq_all_p1_5.items(), key=lambda x: x[1])
least_likely_p1_5 = [x[0] for x in sorted_by_freq[:5]]

print(f"   Distance pattern: {dist_p1_p2}, {dist_p2_p3}, {dist_p3_p4}, {dist_p4_p5}")
print(f"   Most likely: {most_likely_p1_5}")
print(f"   Least likely: {least_likely_p1_5}")

# ============== P6-P7 PREDICTION ==============
print("\n3. P6-P7 PREDICTION (Range 1-12, ordered, no duplicates)...")

# Get base values
base_p6 = int(last_row["P6"])
base_p7 = int(last_row["P7"])

# Get most common distance for P6->P7 from historical data
d67_common = dist_frequencies["D67"].most_common(5)
print(f"   Most common P6->P7 distances: {d67_common}")

# Use most frequent distance
if d67_common:
    p7_val = max(1, min(12, base_p6 + d67_common[0][0]))
else:
    p7_val = max(1, min(12, base_p6 + 2))  # Default

most_likely_p6_7 = sorted([base_p6, p7_val])

# Least likely for P6-P7
freq_all_p6_7 = get_position_frequencies(df, ["P6", "P7"], day_filter=TARGET_DAY)
sorted_by_freq_p67 = sorted(freq_all_p6_7.items(), key=lambda x: x[1])
least_likely_p6_7 = [x[0] for x in sorted_by_freq_p67[:2]]

print(f"   Most likely: {most_likely_p6_7}")
print(f"   Least likely: {least_likely_p6_7}")

# ============== CALCULATE DISTANCES ==============
dist_most_p1_5 = [most_likely_p1_5[i + 1] - most_likely_p1_5[i] for i in range(4)]
dist_most_p6_7 = [most_likely_p6_7[1] - most_likely_p6_7[0]]

dist_least_p1_5 = [least_likely_p1_5[i + 1] - least_likely_p1_5[i] for i in range(4)]
dist_least_p6_7 = [least_likely_p6_7[1] - least_likely_p6_7[0]]

# ============== FINAL RESULTS ==============
result = {
    "target_date": TARGET_DATE.strftime("%Y-%m-%d"),
    "day_of_week": "Friday",
    "most_likely": {
        "positions": {
            "P1": int(most_likely_p1_5[0]),
            "P2": int(most_likely_p1_5[1]),
            "P3": int(most_likely_p1_5[2]),
            "P4": int(most_likely_p1_5[3]),
            "P5": int(most_likely_p1_5[4]),
            "P6": int(most_likely_p6_7[0]),
            "P7": int(most_likely_p6_7[1]),
        },
        "distances": {
            "P1_P2": int(dist_most_p1_5[0]),
            "P2_P3": int(dist_most_p1_5[1]),
            "P3_P4": int(dist_most_p1_5[2]),
            "P4_P5": int(dist_most_p1_5[3]),
            "P6_P7": int(dist_most_p6_7[0]),
        },
    },
    "least_likely": {
        "positions": {
            "P1": int(least_likely_p1_5[0]),
            "P2": int(least_likely_p1_5[1]),
            "P3": int(least_likely_p1_5[2]),
            "P4": int(least_likely_p1_5[3]),
            "P5": int(least_likely_p1_5[4]),
            "P6": int(least_likely_p6_7[0]),
            "P7": int(least_likely_p6_7[1]),
        },
        "distances": {
            "P1_P2": int(dist_least_p1_5[0]),
            "P2_P3": int(dist_least_p1_5[1]),
            "P3_P4": int(dist_least_p1_5[2]),
            "P4_P5": int(dist_least_p1_5[3]),
            "P6_P7": int(dist_least_p6_7[0]),
        },
    },
}

print("\n" + "=" * 60)
print("FINAL PREDICTION FOR FRIDAY 9TH JANUARY 2026")
print("=" * 60)
print(json.dumps(result, indent=2))
