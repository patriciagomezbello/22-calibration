"""
Geoffrey Hinton's Deep Learning Approach for Calibration Prediction
Uses: Distributed Representations, Dropout, ReLU, Deep Belief Networks
Target: Friday 9th January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json


# Set random seed for reproducibility
np.random.seed(42)


# ============== GEOFFREY HINTON'S TECHNIQUES ==============


class DeepBeliefNetwork:
    """Hinton's Deep Belief Network with Restricted Boltzmann Machines"""

    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.rbms = []

        # Initialize RBMs for each layer (Hinton's greedy layer-wise training)
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(
                {
                    "W": np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01,
                    "b": np.zeros(layer_sizes[i + 1]),
                    "c": np.zeros(layer_sizes[i]),
                }
            )

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_rbm(self, rbm, data, epochs=10, k=1):
        """Train single RBM using contrastive divergence (Hinton's method)"""
        n_samples = data.shape[0]

        for epoch in range(epochs):
            # Positive phase
            h_prob = self.sigmoid(np.dot(data, rbm["W"]) + rbm["b"])
            h_state = (h_prob > np.random.rand(n_samples, h_prob.shape[1])).astype(
                float
            )

            # Negative phase (k steps of Gibbs sampling)
            for _ in range(k):
                v_prob = self.sigmoid(np.dot(h_state, rbm["W"].T) + rbm["c"])
                v_state = (v_prob > np.random.rand(n_samples, v_prob.shape[1])).astype(
                    float
                )
                h_prob = self.sigmoid(np.dot(v_state, rbm["W"]) + rbm["b"])
                h_state = (h_prob > np.random.rand(n_samples, h_prob.shape[1])).astype(
                    float
                )

            # Update weights
            rbm["W"] += (
                self.lr
                * (np.dot(data.T, h_prob) - np.dot(v_state.T, h_prob))
                / n_samples
            )
            rbm["b"] += self.lr * (h_prob.mean(axis=0) - h_prob.mean(axis=0))
            rbm["c"] += self.lr * (data.mean(axis=0) - v_prob.mean(axis=0))

    def train(self, data):
        """Greedy layer-wise training (Hinton's method)"""
        current_input = data
        for i, rbm in enumerate(self.rbms):
            self.train_rbm(rbm, current_input, epochs=10)
            # Transform to next layer
            current_input = self.sigmoid(np.dot(current_input, rbm["W"]) + rbm["b"])
        return current_input

    def forward(self, x):
        """Forward pass through DBN"""
        h = x
        for rbm in self.rbms:
            h = self.sigmoid(np.dot(h, rbm["W"]) + rbm["b"])
        return h


class NeuralNetworkWithDropout:
    """Neural Network with Dropout (Hinton's dropout technique)"""

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        # Distributed representations (shared features)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.dropout_rate = dropout_rate

    def relu(self, x):
        """ReLU activation (Hinton's preferred activation)"""
        return np.maximum(0, x)

    def relu_deriv(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)

    def dropout_mask(self, size):
        """Generate dropout mask (inverted dropout for training)"""
        mask = (np.random.rand(*size) > self.dropout_rate) / (1 - self.dropout_rate)
        return mask

    def forward(self, x, training=True):
        """Forward pass with dropout during training only"""
        # Layer 1 with ReLU
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Dropout only during training (Hinton's technique)
        if training:
            self.mask = self.dropout_mask(self.a1.shape)
            self.a1 = self.a1 * self.mask
        else:
            self.mask = None

        # Layer 2 (linear for regression)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, x, y, output, lr=0.01, weight_decay=0.0001):
        """Backpropagation with L2 regularization"""
        m = x.shape[0]

        # Output layer gradient
        dz2 = (output - y) / m

        # L2 regularization
        dW2 = np.dot(self.a1.T, dz2) + weight_decay * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradient
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_deriv(self.z1)

        # Apply dropout mask to gradient
        if self.mask is not None:
            dz1 = dz1 * self.mask

        dW1 = np.dot(x.T, dz1) + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent
        self.W2 -= lr * dW2
        self.b2 -= lr * db2.flatten()
        self.W1 -= lr * dW1
        self.b1 -= lr * db1.flatten()

    def train(self, X, y, epochs=1000, lr=0.01, weight_decay=0.0001):
        """Train with early stopping (Hinton's implicit regularization)"""
        best_loss = float("inf")
        patience = 50
        patience_counter = 0

        for epoch in range(epochs):
            # Forward pass with dropout
            output = self.forward(X, training=True)

            # Backward pass
            self.backward(X, y, output, lr, weight_decay)

            # Calculate loss
            loss = np.mean((output - y) ** 2)

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_loss

    def predict(self, X):
        """Predict without dropout (inference mode)"""
        return self.forward(X, training=False)


# ============== LOAD DATA ==============
df = pd.read_csv("data/22.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

TARGET_DATE = datetime(2026, 1, 9)
print(f"Target Date: {TARGET_DATE.strftime('%Y-%m-%d')} (Friday)")
print("=" * 60)

# Get last known values
last_fri = df[df["Date"] <= "2026-01-05"].iloc[-1]
print(f"Last known data: {last_fri['Date'].strftime('%Y-%m-%d')}")
print(
    f"Positions: P1={last_fri['P1']}, P2={last_fri['P2']}, P3={last_fri['P3']}, P4={last_fri['P4']}, P5={last_fri['P5']}"
)
print(f"Positions: P6={last_fri['P6']}, P7={last_fri['P7']}")

base_values = {
    "P1": int(last_fri["P1"]),
    "P2": int(last_fri["P2"]),
    "P3": int(last_fri["P3"]),
    "P4": int(last_fri["P4"]),
    "P5": int(last_fri["P5"]),
    "P6": int(last_fri["P6"]),
    "P7": int(last_fri["P7"]),
}

# ============== HINTON'S DBN PRETRAINING ==============
print("\n1. Hinton's Deep Belief Network Pretraining...")

# Prepare data for DBN
all_positions = df[["P1", "P2", "P3", "P4", "P5"]].values.astype(float)
all_positions_norm = (all_positions - 1) / 49  # Normalize to [0, 1]

# Train DBN (distributed representations)
dbn = DeepBeliefNetwork(layer_sizes=[5, 10, 5], learning_rate=0.01)
dbn_features = dbn.train(all_positions_norm)
print(f"   DBN features shape: {dbn_features.shape}")

# ============== TRAIN DROPOUT NETWORK ==============
print("\n2. Neural Network with Dropout (Hinton's technique)...")

# Create training data: predict next from current
X_train = all_positions_norm[:-1]
y_train = all_positions_norm[1:]

# Normalize P6-P7
p6_p7_data = df[["P6", "P7"]].values.astype(float)
p6_p7_norm = (p6_p7_data - 1) / 11
X_p67 = p6_p7_norm[:-1]
y_p67 = p6_p7_norm[1:]

# Train P1-P5 network with dropout
nn_p1_5 = NeuralNetworkWithDropout(
    input_size=5, hidden_size=10, output_size=5, dropout_rate=0.5
)
nn_p1_5.train(X_train, y_train, epochs=1000, lr=0.01, weight_decay=0.001)

# Train P6-P7 network with dropout
nn_p6_7 = NeuralNetworkWithDropout(
    input_size=2, hidden_size=5, output_size=2, dropout_rate=0.5
)
nn_p6_7.train(X_p67, y_p67, epochs=1000, lr=0.01, weight_decay=0.001)

print("   Training complete with dropout, early stopping, and L2 regularization")

# ============== PREDICT ==============
print("\n3. Making predictions...")

# User's distance pattern
DIST_P1_P2 = 5
DIST_P2_P3 = 2
DIST_P3_P4 = 11
DIST_P4_P5 = 10

# Get last normalized values
last_norm = (all_positions[-1] - 1) / 49
last_norm_p67 = (p6_p7_data[-1] - 1) / 11

# Predict with DBN features
dbn_pred = dbn.forward(last_norm.reshape(1, -1))
nn_pred = nn_p1_5.predict(last_norm.reshape(1, -1))

# Combine DBN and NN predictions (ensemble)
combined_pred = 0.5 * dbn_pred.flatten() + 0.5 * nn_pred.flatten()
combined_pred = np.clip(combined_pred, 0, 1)
pred_p1_5 = (combined_pred * 49 + 1).astype(int)

# Apply user's distance pattern
p1_next = min(50, base_values["P1"] + DIST_P1_P2)
p2_next = min(50, base_values["P2"] + DIST_P2_P3)
p3_next = min(50, base_values["P3"] + DIST_P3_P4)
p4_next = min(50, base_values["P4"] + DIST_P4_P5)
p5_next = min(50, base_values["P5"] + DIST_P4_P5)

# Blend with NN prediction
final_p1_5 = sorted(
    set(
        [
            p1_next,
            p2_next,
            p3_next,
            p4_next,
            p5_next,
            int(pred_p1_5[0]),
            int(pred_p1_5[1]),
            int(pred_p1_5[2]),
        ]
    )
)[:5]

# P6-P7 prediction
p67_pred = nn_p6_7.predict(last_norm_p67.reshape(1, -1)).flatten()
p67_pred = np.clip(p67_pred, 0, 1)
pred_p6_7 = (p67_pred * 11 + 1).astype(int)

p6_next = base_values["P6"]
p7_next = min(12, base_values["P6"] + (base_values["P7"] - base_values["P6"]))

final_p6_7 = sorted(set([p6_next, p7_next, int(pred_p6_7[0]), int(pred_p6_7[1])]))[:2]

# Least likely (opposite pattern)
least_p1 = max(1, base_values["P1"] - DIST_P1_P2)
least_p2 = max(1, base_values["P2"] - DIST_P2_P3)
least_p3 = max(1, base_values["P3"] - DIST_P3_P4)
least_p4 = max(1, base_values["P4"] - DIST_P4_P5)
least_p5 = max(1, base_values["P5"] - DIST_P4_P5)

least_likely_p1_5 = sorted(set([least_p1, least_p2, least_p3, least_p4, least_p5]))[:5]
least_likely_p6_7 = sorted(
    [max(1, base_values["P6"] - 3), max(1, base_values["P7"] - 3)]
)[:2]

# Calculate distances
dist_most_p1_5 = [final_p1_5[i + 1] - final_p1_5[i] for i in range(4)]
dist_most_p6_7 = [final_p6_7[1] - final_p6_7[0]]
dist_least_p1_5 = [least_likely_p1_5[i + 1] - least_likely_p1_5[i] for i in range(4)]
dist_least_p6_7 = [least_likely_p6_7[1] - least_likely_p6_7[0]]

print(f"\n4. Hinton's Mathematics Applied:")
print(f"   - Deep Belief Network: Greedy layer-wise pretraining")
print(f"   - Dropout: 50% during training, 0% during inference")
print(f"   - ReLU activation: Non-linear distributed representations")
print(f"   - Early stopping: Implicit regularization")
print(f"   - L2 regularization: Weight decay = 0.001")

print(f"\n   MOST LIKELY: P1-P5={final_p1_5}, P6-P7={final_p6_7}")
print(f"   LEAST LIKELY: P1-P5={least_likely_p1_5}, P6-P7={least_likely_p6_7}")

# Final result
result = {
    "target_date": TARGET_DATE.strftime("%Y-%m-%d"),
    "day_of_week": "Friday",
    "hinton_techniques": {
        "distributed_representations": "DBN with 3 layers [5, 10, 5]",
        "dropout": "50% rate during training, inverted dropout",
        "relu": "ReLU activation in hidden layer",
        "early_stopping_patience": 50,
        "l2_regularization": 0.001,
    },
    "most_likely": {
        "positions": {
            "P1": int(final_p1_5[0]),
            "P2": int(final_p1_5[1]),
            "P3": int(final_p1_5[2]),
            "P4": int(final_p1_5[3]),
            "P5": int(final_p1_5[4]),
            "P6": int(final_p6_7[0]),
            "P7": int(final_p6_7[1]),
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
