import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from cnn import CNN

# Use CUDA if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Some values we need
batchSize = 64
classesN = 10
lr = 0.003
epochsN = 35

transformsAll = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.49, 0.48, 0.445],
                                                         std=[0.199, 0.200, 0.201])
                                    ])

# Training data from CIFAR10
dataTrain = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         transform=transformsAll,
                                         download=True)

# Testing data from CIFAR10
dataTest = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=transformsAll,
                                        download=True)

# Instantiate loader objects to facilitate processing
trainingDataLoader = torch.utils.data.DataLoader(dataset=dataTrain,
                                                 batch_size=batchSize,
                                                 shuffle=True)

testingDataLoader = torch.utils.data.DataLoader(dataset=dataTest,
                                                batch_size=batchSize,
                                                shuffle=True)

CNNModel = CNN(classesN)

if torch.cuda.is_available():
    CNNModel.cuda()

crossEntropyLoss = nn.CrossEntropyLoss()
SGDOptimizer = torch.optim.SGD(CNNModel.parameters(), lr=lr, weight_decay=0.00477, momentum=0.75)
stepsN = len(trainingDataLoader)

for currentEpoch in range(epochsN):
    # Training in batches
    for k, (images, labels) in enumerate(trainingDataLoader):
        images = images.to(device)
        labels = labels.to(device)

        # Fw propagation...
        outputs = CNNModel(images)
        currentLoss = crossEntropyLoss(outputs, labels)

        # Bw propagation...
        SGDOptimizer.zero_grad()
        currentLoss.backward()
        SGDOptimizer.step()

    print('Epoch = [{}/{}], Loss = {:.3f}'.format(currentEpoch + 1, epochsN, currentLoss.item()))

with torch.no_grad():
    truePositives = 0
    all = 0
    for images, labels in trainingDataLoader:
        # Save to device
        images = images.to(device)
        labels = labels.to(device)

        # Get outputs from CNN model
        outputs = CNNModel(images)
        _, predicted = torch.max(outputs.data, 1)
        all += labels.size(0)

        # Get true positives
        truePositives += (predicted == labels).sum().item()

    print('Accuracy = {} %'.format(100 * truePositives / all))
