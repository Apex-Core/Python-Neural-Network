import numpy as np 

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# Y = [[18,   -29],
#  [0.14100315, -5],
#  [4, -3]]

Y = [[0.148296,   -0.08397602],
 [0.14100315, -0.01340469],
 [0.20124979, -0.07290616]]

correctValues = [0, 0, 0, 0, 0, 0]
amountCorrect = 0
amountOfVariables = 6;
percentCorrect = 0

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

def compareValues():
    currentIndex = 0

    for i in range(0, 3):
        for j in range(0, 2):
            if layer2.output[i][j] - Y[i][j] <= 0.5 and layer2.output[i][j] - Y[i][j] >= -0.5:
                correctValues[currentIndex] = 1
                currentIndex += 1


compareValues()

amountCorrect = correctValues[0] + correctValues[1] + correctValues[2] + correctValues[3] + correctValues[4] + correctValues[5]
percentCorrect = (amountCorrect / amountOfVariables) * 100

print(percentCorrect)
print(correctValues)