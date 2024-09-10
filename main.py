import src.mnist_loader as mnist_loader
# import src.network_ver1 as network
import src.network_ver2 as network
import os

print("Current working directory:", os.getcwd())

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Set up the Network
net = network.Network([784, 30, 10])

# Train the network using Stochastic Gradient Descent
net.gradient_descent(training_data, 30, 10, 3.0,
                     validation_data=validation_data)