import numpy as np


class NeuralNetworkLayer:
    def __init__(self, prevNodes, currentNodes=784, activationType="sigmoid", batchSize=1):
        # current nodes based on MNIST sizes (28 x 28 = 784)
        self.currentNodes = currentNodes
        self.activationType = activationType
        self.prevNodes = prevNodes
        self.batchSize = batchSize
        self.weights = np.random.randn(self.currentNodes, self.prevNodes) * np.sqrt(2 / self.currentNodes)
        self.biases = np.random.randn(self.currentNodes, 1) * np.sqrt(2 / self.currentNodes)
        self.zMatrix = np.random.rand(self.currentNodes, self.batchSize)
        self.activationMatrix = np.random.rand(self.currentNodes, self.batchSize)
        self.dMatrix = np.zeros(self.activationMatrix.shape)
        self.deltaAccumulatorMatrix = np.zeros(self.weights.shape)

    def activate(self):
        if self.activationType == "relu":
            self.activationMatrix = self.reluActivation(x=self.zMatrix)
        if self.activationType == "sigmoid":
            self.activationMatrix = self.sigmoidActivation(x=self.zMatrix)
        if self.activationType == "softmax":
            self.activationMatrix = self.softmaxActivation(x=self.zMatrix)

    def activateIgnoreOriginal(self, x):
        if self.activationType == "relu":
            return self.reluActivation(x=x, ignoreOriginal=True)
        if self.activationType == "sigmoid":
            return self.sigmoidActivation(x=x, ignoreOriginal=True)
        if self.activationType == "softmax":
            return self.softmaxActivation(x=x, ignoreOriginal=True)

    def reluActivation(self, x, ignoreOriginal=False):
        x[x <= 0] = 0
        if not ignoreOriginal:
            return x
        # Ignoring original activator
        x[x > 0] = 1
        return x

    def sigmoidActivation(self, x, ignoreOriginal=False):
        sigmoid_original = 1 / (1 + np.e ** (-x))
        if not ignoreOriginal:
            return sigmoid_original
        # Ignoring original activator
        return sigmoid_original * (1 - sigmoid_original)

    def softmaxActivation(self, x, ignoreOriginal=False):
        if not ignoreOriginal:
            return (np.e ** x) / sum(np.e ** x)
        # Ignoring original activator
        return (np.e ** x) / sum((np.e ** x)) * (1 - (np.e ** x) / sum((np.e ** x)))
