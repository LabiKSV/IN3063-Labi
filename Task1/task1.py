import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path


#snippet for bcolors from https://www.codegrepper.com/code-examples/python/python+change+print+color
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def gamemenu():
    print('Grid game')
    print('1. Game mode 1 (heuristic)')
    print('2. Game mode 2 (heuristic)')
    print('3. Game mode 1 (Dijkstra)')
    print('4. Game mode 2 (Dijkstra)')
    print('5. Quit game')


def creategrid():
    # grid setup
    global n
    global h
    global w
    global grid
    n = int(input('Enter the value of `n` (grid values random between 0 and n) = '))
    h = int(input('Grid height = '))
    w = int(input('Grid width = '))
    grid = np.random.randint(0, n, size=(h, w))

    # display the random grid
    print("Generated grid: ")
    print(grid)
    print()


def gridwithpath(path):
    path = np.array(path)
    str = ""
    for i in range(0, h):
        for j in range(0, w):
            inPath = False
            for k in range(0, len(path)):
                if i == path[k, 0] and j == path[k, 1]:
                    inPath = True
                    break
            if inPath:
                str += "{} {}{}".format(bcolors.WARNING, grid[i,j], bcolors.ENDC)
            else:
                str += " {}".format(grid[i, j])
        print(str)
        str = ""


def mode1heur():
    # save some details I need
    i = 0
    j = 0
    myPos = grid[i, j]
    myPath = []
    myPosForPath = (i, j)
    myPath.append(myPosForPath)
    totalCost = myPos

    # get to the last cell (bottom-right)
    while (i != h - 1) and (j != w - 1):
        costGoDown = -1
        costGoRight = -1
        if i < h - 1:
            costGoDown = grid[i + 1, j]
        if j < w - 1:
            costGoRight = grid[i, j + 1]

        if costGoDown == -1:
            totalCost += costGoRight
            j = j + 1
            myPos = grid[i, j]
            myPosForPath = (i, j)
            myPath.append(myPosForPath)
        else:
            if costGoRight == -1:
                totalCost += costGoDown
                i = i + 1
                myPos = grid[i, j]
                myPosForPath = (i, j)
                myPath.append(myPosForPath)
            else:
                if costGoRight < costGoDown:
                    totalCost += costGoRight
                    j = j + 1
                    myPos = grid[i, j]
                    myPosForPath = (i, j)
                    myPath.append(myPosForPath)
                else:
                    totalCost += costGoDown
                    i = i + 1
                    myPos = grid[i, j]
                    myPosForPath = (i, j)
                    myPath.append(myPosForPath)

    if i == h - 1:
        for jj in range(j + 1, w):
            myPos = grid[i, jj]
            myPosForPath = (i, jj)
            myPath.append(myPosForPath)
            totalCost += myPos
    elif j == w - 1:
        for ii in range(i + 1, h):
            myPos = grid[ii, j]
            myPosForPath = (ii, j)
            myPath.append(myPosForPath)
            totalCost += myPos

    return totalCost, myPath


def mode2heur():
    # save some details I need
    i = 0
    j = 0
    myPos = grid[i, j]
    myPath = []
    myPosForPath = (i, j)
    myPath.append(myPosForPath)
    totalCost = myPos

    # get to the last cell (bottom-right)
    while (i != h - 1) and (j != w - 1):
        costGoDown = -1
        costGoRight = -1
        if i < h - 1:
            costGoDown = abs(myPos - grid[i + 1, j])
        if j < w - 1:
            costGoRight = abs(myPos - grid[i, j + 1])

        if costGoDown == -1:
            totalCost += costGoRight
            j = j + 1
            myPos = grid[i, j]
            myPosForPath = (i, j)
            myPath.append(myPosForPath)
        else:
            if costGoRight == -1:
                totalCost += costGoDown
                i = i + 1
                myPos = grid[i, j]
                myPosForPath = (i, j)
                myPath.append(myPosForPath)
            else:
                if costGoRight < costGoDown:
                    totalCost += costGoRight
                    j = j + 1
                    myPos = grid[i, j]
                    myPosForPath = (i, j)
                    myPath.append(myPosForPath)
                else:
                    totalCost += costGoDown
                    i = i + 1
                    myPos = grid[i, j]
                    myPosForPath = (i, j)
                    myPath.append(myPosForPath)

    if i == h - 1:
        for jj in range(j + 1, w):
            myPosForPath = (i, jj)
            myPath.append(myPosForPath)
            totalCost += abs(myPos - grid[i, jj])
            myPos = grid[i, jj]
    elif j == w - 1:
        for ii in range(i + 1, h):
            myPosForPath = (ii, j)
            myPath.append(myPosForPath)
            totalCost += abs(myPos - grid[ii, j])
            myPos = grid[ii, j]

    return totalCost, myPath


def showpath(path):
    # display all cells as x,y pairs and -> between them
    for row in path[:-1]:
        print(row, end=" -> ")
    print(path[-1])


while True:
    #print(f"{bcolors.WARNING}Error : Test message !{bcolors.ENDC}")
    gamemenu()
    userChoice = input('Your selection = ')

    # Game mode 1 (heuristic)
    if userChoice == '1':
        creategrid()
        costAndPath = mode1heur()
        print('Mode 1 cost = ', costAndPath[0])
        print('Mode 1 path:')
        showpath(costAndPath[1])
        break

    # Game mode 2 (heuristic)
    elif userChoice == '2':
        creategrid()
        costAndPath = mode2heur()
        print('Mode 2 cost = ', costAndPath[0])
        print('Mode 2 path:')
        showpath(costAndPath[1])
        print('Mode 2 grid with path:')
        gridwithpath(costAndPath[1])
        break

    # Game mode 1 (Dijkstra)
    elif userChoice == '3':
        creategrid()
        break

    # Game mode 2 (Dijkstra)
    elif userChoice == '4':
        creategrid()
        break

    # User quit game
    elif userChoice == '5':
        print('Quit game! Done.')
        break