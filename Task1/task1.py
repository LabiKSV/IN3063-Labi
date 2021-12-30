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
    graph = [[100000] * h for _ in range(w)]
    for i in range(0, h-1):
        for j in range(0, w-1):
            graph[i, j] = 0


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