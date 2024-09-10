import src.mnist_loader as mnist_loader
import src.network_ver1 as network1
import src.network_ver2 as network2
import src.network_ver3 as network3
import os

print("Current working directory:", os.getcwd())

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Set up the Network
net = network3.Network([784, 30, 10], cost=network3.CrossEntropyCost, init_method='efficient')

# Train the network using Stochastic Gradient Descent
net.gradient_descent(training_data, 50, 10, 3.0,
                     validation_data=validation_data)