import numpy as np 

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

class LayerDense:
    def __init__ (self, inputCount, neronCount):
        self.weights = 0.10 * np.random.randn(inputCount, neronCount)
        self.biases = np.zeros((1, neronCount))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)