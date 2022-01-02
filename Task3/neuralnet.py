import numpy as np
import matplotlib.pyplot as plt
import helpers
from neuralnetworklayer import NeuralNetworkLayer


class NeuralNetwork:
    def __init__(self, x, y, lr=0.002, reg=1, batchSize=1, stopEarly=False, lrScaling=0, minLR=0):
        self.a0 = None
        self.hx = None
        self.x = x
        self.y = y
        self.reg = reg
        self.minLR = minLR
        self.stopEarly = stopEarly
        self.lrScaling = lrScaling
        self.batchSize = batchSize
        self.lr = lr
        self.layers = []
        self.histLoss = []
        self.histAccuracy = []
        # Features and data points
        self.countFeatures = self.x.shape[0]
        self.countDataPoints = self.x.shape[1]
        if self.batchSize <= 0:
            self.batchSize = self.countDataPoints

    def score(self, x_test, y_test):
        predictions = self.predict(x_test)
        correctAnsN = 0
        testsN = len(predictions[0])
        for i in range(testsN):
            prediction = helpers.mydecode(helpers.myprobabilityencode(predictions[:, i]))
            correct = helpers.mydecode(y_test[:, i])
            if prediction == correct:
                correctAnsN += 1
        accuracy = correctAnsN / testsN
        return accuracy

    def predict(self, x_test):
        self.fwPass(x_test, testSize=len(x_test))
        return self.hx

    def addLayer(self, nodes=784, activationType="sigmoid"):
        nLayers = len(self.layers)
        # Check node counts based on which layer this is
        nodeCount = self.countFeatures
        if nLayers != 0:
            nodeCount = self.layers[nLayers - 1].currentNodes
        # Add new layer
        newLayer = NeuralNetworkLayer(prevNodes=nodeCount, currentNodes=nodes, activationType=activationType,
                                      batchSize=self.batchSize)
        self.layers.append(newLayer)

    def fwPass(self, x, testSize=0):
        # Each layer will have the test size
        if testSize > 0:
            for i in range(len(self.layers)):
                self.layers[i].batchSize = testSize

        # Activate all layers (forward pass)
        layer = self.layers[0]
        self.a0 = x
        layer.zMatrix = layer.weights.dot(self.a0) + layer.biases
        layer.activate()
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            layer.zMatrix = layer.weights.dot(prev_layer.activationMatrix) + layer.biases
            layer.activate()
        self.hx = layer.activationMatrix

    def bwPass(self, x, y):
        # Go from last layer backwards
        lastLayerPos = len(self.layers) - 1
        i = lastLayerPos
        layer = self.layers[i]
        layer.dMatrix = self.hx - y
        i -= 1
        while i >= 0:
            layer = self.layers[i]
            nextLayer = self.layers[i + 1]
            layer.dMatrix = (nextLayer.weights.transpose().dot(nextLayer.dMatrix)) * layer.activateIgnoreOriginal(
                layer.zMatrix)
            # Decrement
            i -= 1

        # Update layer weights in all layers
        # First layer
        if self.layers >= 1:
            layer = self.layers[0]
            self.a0 = x
            layer.deltaAccumulatorMatrix += layer.dMatrix.dot(self.a0.transpose())
            gradients = layer.deltaAccumulatorMatrix * (self.reg / self.countDataPoints)
            layer.weights -= gradients * self.lr

        # All other layers
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.deltaAccumulatorMatrix += layer.dMatrix.dot(self.layers[i - 1].activationMatrix.transpose())
            gradients = layer.deltaAccumulatorMatrix * (self.reg / self.countDataPoints)
            layer.weights -= gradients * self.lr

    def getLoss(self, hx, y):
        return (sum(sum((y * np.log(hx)) + ((1 - y) * np.log(1 - hx)))) / self.countDataPoints) * -1

    def fit(self, epochsN, calcAccuracy=None, lossPrec=10):
        nBatches = int(self.countDataPoints / self.batchSize)
        loss = 0
        for currentEpoch in range(epochsN):
            print(f"Now at -> Epoch {currentEpoch + 1}/{epochsN}. ", end="")
            for b in range(nBatches):
                start = b * self.batchSize
                end = (b + 1) * self.batchSize
                xi = self.x[:, start:end]
                yi = self.y[:, start:end]
                self.fwPass(xi)
                self.bwPass(xi, yi)
            loss = self.getLoss(self.hx, yi)
            print(f"\nLoss = {np.around(loss, lossPrec)}. ", end="")

            if calcAccuracy is not None:
                x_test = calcAccuracy["x_test"]
                y_test = calcAccuracy["y_test"]
                accuracy = self.score(x_test, y_test)
                self.histAccuracy.append(accuracy)
                print(f"Accuracy = {accuracy}")

            print("")
            self.histLoss.append(loss)
            if currentEpoch > 0 and (self.stopEarly or self.lrScaling > 0):
                if loss > self.histLoss[currentEpoch - 1]:
                    if self.lrScaling > 0:
                        print(f"Learning rate = {self.lr * self.lrScaling}")
                        if self.minLR > 0:
                            self.lr = max((self.lrScaling * self.lr), self.minLR)
                        else:
                            self.lr *= self.lrScaling
                    # If early stopping is set
                    if self.stopEarly:
                        if self.lrScaling <= 0 or self.lr == self.minLR:
                            print(f"Stopped early (epoch {currentEpoch}).")
                            break

    def lossPlot(self, minY=None, maxY=None):
        hist = self.histLoss
        if minY is None:
            plt.ylim((min(hist)), (max(hist)))
        else:
            plt.ylim(minY, maxY)
        plt.title("Loss / Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(list(range(len(hist))), hist)
        plt.show()

    def accuracyPlot(self, minY=None, maxY=None):
        hist = self.histAccuracy
        if len(hist) == 0:
            print("Unknown accuracy...")
            return
        if minY is None:
            plt.ylim((min(hist)), (max(hist)))
        else:
            plt.ylim(minY, maxY)
        plt.title("Accuracy / Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(list(range(len(hist))), hist)
        plt.show()
