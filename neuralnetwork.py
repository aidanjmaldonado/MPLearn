import numpy as np
from math import e, pi

"""
Activation functions:
sigmoid()
    returns the sigmoid of x

ReLU()
    returns the max{0, x}
    
sign()
    returns the sign of the input [-, +] ?
    
threshold()
    returns the activation 1[x > 0] (essentially a boolean of whether x is > 0 or not)
    
tahn? tan? tanh?()
    returns something tan :3
"""

class NeuralNetwork:
    def __init__(self, hiddenLayers: int, activation_func):
        self.hiddenLayers = hiddenLayers
        self.input = Layer
        self.activation_function = self.activation_function

class Layer(NeuralNetwork):
    def __init__self(self, neurons: int, type: str):
        self.neurons = neurons
        self.type = type
        self.types = {"FullyConnected"}

class Neuron(Layer):
    def __init__(self, activation_function, input, output):
        self.activation_function = activation_function
        self.input = input
        self.output = output



class Activation:
    def __init__(self, input):
        self.input = input
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))
    def ReLU(self, input):
        return np.maximum(0, input)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)
