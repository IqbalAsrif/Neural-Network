"""
Network Version 3 Updates:
Added Cross-Entropy Cost Function:
Enhances learning speed by providing better gradient performance, especially when error rates are high.
New Weight Initialization Strategy:
Implemented a more efficient method using a tighter Gaussian distribution, which minimizes covariance between weights.
"""

import random
import time
import numpy as np


class QuadraticCost(object):
    """
    Quadratic Cost Function - Mean Squared Error (1/2n * Sigma(a - y)^2)
    """
    @staticmethod
    def function(a, y):
        return 0.5 / len(a) * np.sum((a - y) ** 2)

    @staticmethod
    def sigma(z, a, y):
        return (a - y) * sigmoid_derivative(z)


class CrossEntropyCost(object):
    """
    Cross Entropy Cost - -Sigma(y ln a + (1 - y) ln (1 - a))
    """
    @staticmethod
    def function(a, y):
        return -1 * np.sum(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))

    @staticmethod
    def sigma(z, a, y):
        return a - y


# Regularization hasn't been included in other functions yet
class Regularization(object):
    @staticmethod
    def L1(weights, param):
        return np.sum(param * np.abs(weights))

    @staticmethod
    def L2(weights, param):
        return np.sum(param * np.square(weights))


class Network(object):
    def __init__(self, sizes, cost=QuadraticCost, init_method='standard', regularization=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.regularization = regularization

        if init_method == 'standard':
            self.standard_gaussian_initialization()
        elif init_method == 'efficient':
            self.efficient_gaussian_initialization()
        else:
            raise ValueError('Initialization Method Unsupported')

    def standard_gaussian_initialization(self):
        self.weights = [np.random.randn(j, i)
                        for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(i, 1) for i in self.sizes[1:]]

    def efficient_gaussian_initialization(self):
        self.weights = [np.random.randn(j, i) / np.sqrt(i)
                        for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(i, 1) for i in self.sizes[1:]]

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
        delta = self.cost.sigma(preactivation_list[-1], activation_list[-1], y)
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
