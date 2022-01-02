import torch
import torchvision
import torchvision.transforms as transforms
import helpers
from neuralnet import NeuralNetwork

# Grab data from MNIST and use download = True
transform = transforms.Compose([transforms.ToTensor()])
mnistTraining = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnistTrainingDataLoader = torch.utils.data.DataLoader(mnistTraining, batch_size=4, shuffle=True, num_workers=2)
mnistTest = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnistTestDataLoader = torch.utils.data.DataLoader(mnistTest, batch_size=4, shuffle=False, num_workers=2)

# Show details
print(f"\nMNIST Training set size = {len(mnistTraining)}")
print(f"MNIST Test set size = {len(mnistTest)}\n")

# Get feature and target arrays
f, t = helpers.xypreprocess(x=mnistTraining.data, y=mnistTraining.targets)
X_test, y_test = helpers.xypreprocess(x=mnistTest.data, y=mnistTest.targets)
fReduced, tReduced = helpers.reducelength(f, t, first=0, last=512)

# Build neural net with 2 sigmoid + 1 relu + 2 sigmoid + softmax as its layers
# Meaning we use both fw and bw propagation
neuralNet = NeuralNetwork(x=fReduced, y=tReduced, lr=1e-6, reg=1, batchSize=256, lrScaling=0.5)
neuralNet.addLayer(nodes=60, activationType="sigmoid")
neuralNet.addLayer(nodes=60, activationType="sigmoid")
neuralNet.addLayer(nodes=60, activationType="relu")
neuralNet.addLayer(nodes=60, activationType="sigmoid")
neuralNet.addLayer(nodes=60, activationType="sigmoid")
neuralNet.addLayer(nodes=23, activationType="softmax")

# Start fitting the model then plot loss and accuracy
neuralNet.fit(epochsN=1000, calcAccuracy={"X_test": X_test, "y_test": y_test})
neuralNet.lossPlot()
accuracy = neuralNet.score(X_test, y_test)
neuralNet.accuracyPlot()

# Display accuracy on screen (besides already being plotted above)
print("Accuracy:")
print(accuracy)
