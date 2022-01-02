import numpy as np


def myencode(input):
    # Encodes an integer into an array of 0s and 1s
    res = np.zeros(10)
    res[input] = 1
    return res


def mydecode(input):
    # Decodes the above encoded array
    res = 0
    for i in range(len(input)):
        if input[i] == 1:
            res = i
    return res


def myprobabilityencode(input):
    # Encode probabilities (1, 0)
    pMax = 0
    pos = 0

    for i in range(len(input)):
        if input[i] > pMax:
            pMax = input[i]
            pos = i

    res = np.zeros(input.shape)
    res[pos] = 1
    return res


def xypreprocess(x, y, applyTranspose=True):
    xx = len(x)
    yy = len(x[0]) ** 2
    x = x.numpy().reshape(xx, yy)
    yEnc = []
    for i in range(len(y)):
        # Apply custom encoding
        valEnc = myencode(y[i])
        yEnc.append(valEnc)
    y = yEnc

    # Create arrays
    x = np.array(x)
    y = np.array(y)

    # Transpose
    if applyTranspose:
        x = x.transpose()
        y = y.transpose()

    # Target and feature arrays from x and y inputs
    return x, y


def reducelength(x, y, first, last):
    f = x[:, first:last]
    t = y[:, first:last]
    return f, t
