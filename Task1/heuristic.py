def heuristicpathsolver(grid, h, w, isMode1=True):
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
            if isMode1:
                costGoDown = grid[i + 1, j]
            else:
                costGoDown = abs(myPos - grid[i + 1, j])
        if j < w - 1:
            if isMode1:
                costGoRight = grid[i, j + 1]
            else:
                costGoRight = abs(myPos - grid[i, j + 1])

        if costGoDown == -1:
            # I must go right, cannot go down
            totalCost += costGoRight
            j = j + 1
            myPos = grid[i, j]
            myPosForPath = (i, j)
            myPath.append(myPosForPath)
        else:
            if costGoRight == -1:
                # I must go down as I cannot go right
                totalCost += costGoDown
                i = i + 1
                myPos = grid[i, j]
                myPosForPath = (i, j)
                myPath.append(myPosForPath)
            else:
                # I could go down or right, so need to check lowest cost (greedy algorithm)
                if costGoRight < costGoDown:
                    # going right is cost-effective
                    totalCost += costGoRight
                    j = j + 1
                    myPos = grid[i, j]
                    myPosForPath = (i, j)
                    myPath.append(myPosForPath)
                else:
                    # going down is cost-effective
                    totalCost += costGoDown
                    i = i + 1
                    myPos = grid[i, j]
                    myPosForPath = (i, j)
                    myPath.append(myPosForPath)

    # if I reached here, if I'm on the last row, must go all the way to the right
    if i == h - 1:
        for jj in range(j + 1, w):
            myPosForPath = (i, jj)
            myPath.append(myPosForPath)
            if isMode1:
                myPos = grid[i, jj]
                totalCost += myPos
            else:
                totalCost += abs(myPos - grid[i, jj])
                myPos = grid[i, jj]
    # if I reached here, if I'm on the last column, must go all the way down to the last cell
    elif j == w - 1:
        for ii in range(i + 1, h):
            myPosForPath = (ii, j)
            myPath.append(myPosForPath)
            if isMode1:
                myPos = grid[ii, j]
                totalCost += myPos
            else:
                totalCost += abs(myPos - grid[ii, j])
                myPos = grid[ii, j]

    return totalCost, myPath


def displayresults(costAndPath):
    print('Mode 1 cost = ', costAndPath[0])
    print('Mode 1 path:')
    showpath(costAndPath[1])
    print('Mode 1 grid with path:')
    gridwithpath(costAndPath[1])
