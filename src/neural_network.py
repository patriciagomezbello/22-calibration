import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases for two layers
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, regression=True):
        # Layer 1: Linear + ReLU (nonlinearity)
        self.Z1 = (
            np.dot(X, self.W1) + self.b1
        )  # Matrix multiplication and vector addition
        self.A1 = self.relu(self.Z1)
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
        # Update weights (gradient descent)
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000, regression=True):
        for epoch in range(epochs):
            output = self.forward(X, regression)
            self.backward(X, y, output, regression)
            if epoch % 100 == 0:
                if regression:
                    loss = np.mean((output - y) ** 2)
                else:
                    loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X, regression=True):
        output = self.forward(X, regression)
        if regression:
            return output
        else:
            return np.argmax(output, axis=1)
