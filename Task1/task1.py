import numpy as np
import heuristic
import display
import matplotlib.pyplot as plt
import matplotlib.path


def gamemenu():
    print('Grid game')
    print('1. Game mode 1 (heuristic)')
    print('2. Game mode 2 (heuristic)')
    print('3. Game mode 1 (Dijkstra)')
    print('4. Game mode 2 (Dijkstra)')
    print('5. Comparison mode (heuristic) - run both modes on same grid')
    print('6. Quit game')


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


def mode1dijkstra():
    # first, I need to transform the entire grid to a weighted graph
    # init to large number to represent infinity (height * width)
    verticesN = h * w  # each cell must be a vertex
    # distances to each cell
    dist = [[10000] * verticesN]
    dist = np.array(dist)
    dist[0] = grid[0, 0]  # this is my initial cost
    # which cell I came from (to each cell)
    prev = [[10000] * verticesN]
    prev = np.array(prev)
    prev[0] = -1  # I mark it as -1 so that means the first cell is my source
    graph = [[10000] * verticesN for _ in range(verticesN)]
    graph = np.array(graph)
    # first loop through all cells, except first row, last row
    for i in range(1, h - 1):
        # cover first cell in row
        currentVertex = i * w
        vRightNeighbor = currentVertex + 1
        vUpNeighbor = currentVertex - w
        vDownNeighbor = currentVertex + w
        graph[vRightNeighbor, currentVertex] = grid[i, 0]
        graph[currentVertex, vRightNeighbor] = grid[i, 1]
        graph[vUpNeighbor, currentVertex] = grid[i, 0]
        graph[currentVertex, vUpNeighbor] = grid[i - 1, 0]
        graph[vDownNeighbor, currentVertex] = grid[i, 0]
        graph[currentVertex, vDownNeighbor] = grid[i + 1, 0]
        for j in range(1, w - 1):
            currentVertex = i * w + j
            vLeftNeighbor = currentVertex - 1
            vRightNeighbor = currentVertex + 1
            vUpNeighbor = currentVertex - w
            vDownNeighbor = currentVertex + w
            graph[vLeftNeighbor, currentVertex] = grid[i, j]
            graph[currentVertex, vLeftNeighbor] = grid[i, j - 1]
            graph[vRightNeighbor, currentVertex] = grid[i, j]
            graph[currentVertex, vRightNeighbor] = grid[i, j + 1]
            graph[vUpNeighbor, currentVertex] = grid[i, j]
            graph[currentVertex, vUpNeighbor] = grid[i - 1, j]
            graph[vDownNeighbor, currentVertex] = grid[i, j]
            graph[currentVertex, vDownNeighbor] = grid[i + 1, j]
        # cover first cell in row
        currentVertex = i * w + w - 1
        vLeftNeighbor = currentVertex - 1
        vUpNeighbor = currentVertex - w
        vDownNeighbor = currentVertex + w
        graph[vLeftNeighbor, currentVertex] = grid[i, w - 1]
        graph[currentVertex, vLeftNeighbor] = grid[i, w - 2]
        graph[vUpNeighbor, currentVertex] = grid[i, w - 1]
        graph[currentVertex, vUpNeighbor] = grid[i - 1, w - 1]
        graph[vDownNeighbor, currentVertex] = grid[i, w - 1]
        graph[currentVertex, vDownNeighbor] = grid[i + 1, w - 1]

    # now first row
    graph[0, 0] = grid[0, 0]
    graph[w - 2, w - 1] = graph[w - 1, w - 2] = grid[0, w - 1]
    j = 1
    while j < w - 1:
        currentVertex = j
        vLeftNeighbor = j - 1
        vRightNeighbor = j + 1
        graph[vLeftNeighbor, currentVertex] = grid[0, j]
        graph[currentVertex, vLeftNeighbor] = grid[0, j - 1]
        graph[vRightNeighbor, currentVertex] = grid[0, j]
        graph[currentVertex, vRightNeighbor] = grid[0, j + 1]
        if h >= 2:
            vDownNeighbor = w + j
            graph[vDownNeighbor, currentVertex] = grid[0, j]
            graph[currentVertex, vDownNeighbor] = grid[1, j]
        j += 1
    # cover last cell too
    currentVertex = j
    vLeftNeighbor = j - 1
    graph[vLeftNeighbor, currentVertex] = grid[0, j]
    graph[currentVertex, vLeftNeighbor] = grid[0, j - 1]
    if h >= 2:
        vDownNeighbor = w + j
        graph[vDownNeighbor, currentVertex] = grid[0, j]
        graph[currentVertex, vDownNeighbor] = grid[1, j]

    # now last row
    if h > 1:
        graph[(h - 2) * w, (h - 1) * w] = grid[h - 1, 0]
        graph[(h - 1) * w, (h - 2) * w] = grid[h - 2, 0]
        graph[(h - 1) * w + w - 2, (h - 1) * w + w - 1] = grid[h - 1, w - 1]
        graph[(h - 1) * w + w - 1, (h - 1) * w + w - 2] = grid[h - 1, w - 2]
        j = 1
        while j < w - 1:
            currentVertex = (h - 1) * w + j
            vLeftNeighbor = currentVertex - 1
            vRightNeighbor = currentVertex + 1
            graph[vLeftNeighbor, currentVertex] = grid[h - 1, j]
            graph[currentVertex, vLeftNeighbor] = grid[h - 1, j - 1]
            graph[vRightNeighbor, currentVertex] = grid[h - 1, j]
            graph[currentVertex, vRightNeighbor] = grid[h - 1, j + 1]
            vUpNeighbor = currentVertex - w
            graph[vUpNeighbor, currentVertex] = grid[h - 1, j]
            graph[currentVertex, vUpNeighbor] = grid[h - 2, j]
            j += 1
        # cover last cell too
        currentVertex = (h - 1) * w + j
        vLeftNeighbor = currentVertex - 1
        graph[vLeftNeighbor, currentVertex] = grid[h - 1, j]
        graph[currentVertex, vLeftNeighbor] = grid[h - 1, j - 1]
        vUpNeighbor = currentVertex - w
        graph[vUpNeighbor, currentVertex] = grid[h - 1, j]
        graph[currentVertex, vUpNeighbor] = grid[h - 2, j]

    print("Dijkstra graph: ")
    print(graph)


while True:
    gamemenu()
    userChoice = input('Your selection = ')

    # Game mode 1 (heuristic)
    if userChoice == '1':
        creategrid()
        costAndPath = heuristic.heuristicpathsolver(grid, h, w, True)
        display.displaycostandpath(grid, h, w, costAndPath, 1)
        break

    # Game mode 2 (heuristic)
    elif userChoice == '2':
        creategrid()
        costAndPath = heuristic.heuristicpathsolver(grid, h, w, False)
        display.displaycostandpath(grid, h, w, costAndPath, 2)
        break

    # Game mode 1 (Dijkstra)
    elif userChoice == '3':
        creategrid()
        mode1dijkstra()
        break

    # Game mode 2 (Dijkstra)
    elif userChoice == '4':
        creategrid()
        break

    # User quit game
    elif userChoice == '5':
        creategrid()
        # Mode 1 first
        costAndPath = heuristic.heuristicpathsolver(grid, h, w, True)
        display.displaycostandpath(grid, h, w, costAndPath, 1)
        # Then mode 2
        costAndPath = heuristic.heuristicpathsolver(grid, h, w, False)
        display.displaycostandpath(grid, h, w, costAndPath, 2)
        break

    # User quit game
    elif userChoice == '6':
        print('Quit game! Done.')
        break
