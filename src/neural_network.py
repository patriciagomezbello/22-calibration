import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LinearRegression


class DBN:
    def __init__(self, layers, learning_rate=0.01, n_iter=10):
        self.rbms = []
        for i in range(len(layers) - 1):
            self.rbms.append(
                BernoulliRBM(
                    n_components=layers[i + 1],
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    random_state=0,
                )
            )
        self.regressor = LinearRegression()

    def train(self, X):
        input_data = X
        for rbm in self.rbms:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)
        # Train regressor on final layer
        self.regressor.fit(input_data, X)  # Autoencoder style

    def predict(self, X):
        input_data = X
        for rbm in self.rbms:
            input_data = rbm.transform(input_data)
        return self.regressor.predict(input_data)


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.01,
        dropout_rate=0.2,
        weight_decay=0.0001,
    ):
        # Initialize weights and biases for two layers
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, regression=True, training=True):
        # Noise injection during training
        if training:
            noise = np.random.normal(0, 0.2, X.shape)
            X = X + noise
        # Layer 1: Linear + ReLU (nonlinearity)
        self.Z1 = (
            np.dot(X, self.W1) + self.b1
        )  # Matrix multiplication and vector addition
        self.A1 = self.relu(self.Z1)
        if training:
            # Dropout
            dropout_mask = np.random.rand(*self.A1.shape) > self.dropout_rate
            self.A1 *= dropout_mask / (1 - self.dropout_rate)
        # Layer 2: Linear
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        if regression:
            self.A2 = self.Z2  # For regression, no activation
        else:
            self.A2 = self.softmax(self.Z2)  # For probability distribution
        return self.A2

    def backward(self, X, y, output, regression=True):
        m = X.shape[0]
        # Compute gradients for output layer
        if regression:
            dZ2 = output - y  # MSE
        else:
            dZ2 = output - y  # Cross-entropy with softmax
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        # Backprop to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        # Update weights (gradient descent with L2 regularization)
        self.W2 -= self.learning_rate * (dW2 + self.weight_decay * self.W2)
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * (dW1 + self.weight_decay * self.W1)
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000, regression=True, early_stopping_patience=50):
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            output = self.forward(X, regression)
            self.backward(X, y, output, regression)
            if regression:
                loss = np.mean((output - y) ** 2)
            else:
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, X, regression=True):
        output = self.forward(X, regression, training=False)
        if regression:
            return output
        else:
            return np.argmax(output, axis=1)
