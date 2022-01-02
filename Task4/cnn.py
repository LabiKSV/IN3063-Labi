import torch.nn as nn


# Creating a CNN class
class CNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def fw(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max2(out)

        out = out.reshape(out.size(0), -1)

        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)

        return out
