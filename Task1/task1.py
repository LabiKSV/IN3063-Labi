import numpy as np


def showpath(path):
    for row in path[:-1]:
        print(row, end=" -> ")
    print(path[-1])


# grid setup
n = 10
h = 11
w = 15
grid = np.random.randint(0, n, size=(h, w))

print("Generated grid: ")
print(grid)
print()

# initial testing for display method
path = [[1, 2], [3, 4]]

showpath(path)