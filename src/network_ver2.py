"""
Revising Network Ver 1
Implementing Full Matrix-Based for Stochastic Gradient Descent
Taking advantage of linear algebra library like NumPy
"""

import random
import time
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(j, i)
                        for i, j in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]

    def gradient_descent(self, training_data, epochs, mini_batch_size,
                         learning_rate, validation_data=None):
        if validation_data: n_valid = len(validation_data)
        n = len(training_data)
        for j in range(epochs):
            time_start = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            time_end = time.time()

            if validation_data:
                print("Epoch {}: {} / {}, took {:.2f} seconds"
                      .format(j, self.evaluate(validation_data), n_valid,
                              time_end - time_start))
            else:
                print("Epoch {} complete in {:.2f} seconds"
                      .format(j, time_end - time_start))

    def update_mini_batch(self, mini_batch, learning_rate):
        # Mini Batch are done simultaneously, not in loop anymore
        X = np.array([x.ravel() for x, y in mini_batch]).transpose()
        Y = np.array([y.ravel() for x, y in mini_batch]).transpose()

        gradient_w, gradient_b = self.backward_propagation(X, Y)

        self.weights = [w - learning_rate / len(mini_batch) * gw
                        for w, gw in zip(self.weights, gradient_w)]
        self.biases = [b - learning_rate / len(mini_batch) * gb
                       for b, gb in zip(self.biases, gradient_b)]

    def backward_propagation(self, x, y):
        # List Initialization for Change in Weight and Biases
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        # Forward Pass
        activation = x
        activation_list = [x]
        preactivation_list = []

        for w, b in zip(self.weights, self.biases):
            preactivation = np.dot(w, activation) + b
            preactivation_list.append(preactivation)
            activation = sigmoid(preactivation)
            activation_list.append(activation)

        # Backward Pass
        # Last Layer
        delta = self.cost_derivative(activation_list[-1], y) * sigmoid_derivative(preactivation_list[-1])
        gradient_w[-1] = np.dot(delta, activation_list[-2].transpose())
        gradient_b[-1] = np.sum(delta, axis=1, keepdims=True)

        # Other Layers
        for layer in range(2, len(self.sizes)):
            preactivation = preactivation_list[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_derivative(preactivation)
            gradient_w[-layer] = np.dot(delta, activation_list[-layer - 1].transpose())
            gradient_b[-layer] = np.sum(delta, axis=1, keepdims=True)

        return gradient_w, gradient_b

    def run_network(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.run_network(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output, y):
        """
        Cost Function - Mean Squared Error (1/2n * Sigma(a - y)^2)
        """
        return output - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp((-z)))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
