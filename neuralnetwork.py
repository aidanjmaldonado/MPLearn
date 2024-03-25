import numpy as np
from math import e, pi

class NeuralNetwork:
    def __init__(self, hiddenLayers: int):
        self.hiddenLayers = hiddenLayers
        self.input = Layer
`
class Layer(NeuralNetwork):
    def __init__self(self, neruons: int):
        self.neurons = neurons






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
