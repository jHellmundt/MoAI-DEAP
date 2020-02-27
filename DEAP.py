# %matplotlib inline
from deap import creator, base, tools, gp, algorithms
import matplotlib.pyplot as plt
import random
import operator
import numpy as np

creator.create("FitnessMax", base.Fitness, weights = (-1.0,))
creator.create("Individual", np.ndarray, fitness = creator.FitnessMax)

HEIGHT = 5
WIDTH = 5
COLORS = 3


def init2d(icls, shape):
    return icls(np.random.randint(1, COLORS + 1, shape))


toolbox = base.Toolbox()
toolbox.register("individual", init2d, creator.Individual, shape = (HEIGHT, WIDTH))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def update_neighbours(neigh1, neigh2):
    for color in neigh2:
        for neigh in neigh2[color]:
            if color in neigh1:
                if neigh not in neigh1[color]:
                    neigh1.setdefault(color, []).append(neigh)
            else:
                neigh1.setdefault(color, []).append(neigh)
    return neigh1


def color_fields(individual):
    counts = np.zeros(individual.shape, dtype = int) + 1
    neighbours = np.frompyfunc(dict, 0, 1)(np.empty((individual.shape[0],individual.shape[1]), dtype = object))
    fields = dict()
    conflicts = dict()
    for i in range(1, COLORS + 1):
        conflicts[i] = []
        fields[i] = []
    # Count the number of pixels of the same color in each field
    for row in range(0, individual.shape[0]):
        for col in range(0, individual.shape[1]):
            color = individual[row, col]
            change = 0  # 0 = unchanged, 1 = horizontal, 2 = vertical
            if counts[row, col] != 0:
                next_row = row + 1 < individual.shape[0]
                next_col = col + 1 < individual.shape[1]
                # Get all neighbours
                if next_row and color != individual[row + 1, col]:
                    neighbours[row, col].setdefault(individual[row + 1, col], []).append((row + 1, col))
                if next_col and color != individual[row, col + 1]:
                    neighbours[row, col].setdefault(individual[row, col + 1], []).append((row, col + 1))
                if row - 1 >= 0 and color != individual[row - 1, col]:
                    neighbours[row, col].setdefault(individual[row - 1, col], []).append((row - 1, col))
                if col - 1 >= 0 and color != individual[row, col - 1]:
                    neighbours[row, col].setdefault(individual[row, col - 1], []).append((row, col - 1))
                # Move horizontal and create a conflict with the vertical field that can later be resolved
                if next_row and next_col and color == individual[row + 1, col] and color == individual[row, col + 1]:
                    counts[row + 1, col] += counts[row, col]
                    neighbours[row + 1, col] = update_neighbours(neighbours[row + 1, col], neighbours[row, col])
                    counts[row, col] = 0
                    conflicts[color] += [((row + 1, col), (row, col + 1))]
                    change = 1
                # Move horizontal
                elif next_row and color == individual[row + 1, col]:
                    counts[row + 1, col] += counts[row, col]
                    neighbours[row + 1, col] = update_neighbours(neighbours[row + 1, col], neighbours[row, col])
                    counts[row, col] = 0
                    change = 1
                # Move vertical
                elif next_col and color == individual[row, col + 1]:
                    counts[row, col + 1] += counts[row, col]
                    neighbours[row, col + 1] = update_neighbours(neighbours[row, col + 1], neighbours[row, col])
                    counts[row, col] = 0
                    change = 2
                # Change conflict field if the current field already had a conflict (Move conflict)
                if change == 1:
                    conflicts[color] = [conf if conf[0] != (row, col) else ((row + 1, col), conf[1]) for conf in
                                        conflicts[color]]
                    conflicts[color] = [conf if conf[1] != (row, col) else (conf[0], (row + 1, col)) for conf in
                                        conflicts[color]]
                elif change == 2:
                    conflicts[color] = [conf if conf[0] != (row, col) else ((row, col + 1), conf[1]) for conf in
                                        conflicts[color]]
                    conflicts[color] = [conf if conf[1] != (row, col) else (conf[0], (row, col + 1)) for conf in
                                        conflicts[color]]
    # Resolve conflicts
    for color in conflicts:
        for conf in conflicts[color]:
            if conf[0] != conf[1]:
                counts[conf[0][0], conf[0][1]] += counts[conf[1][0], conf[1][1]]
                neighbours[conf[0][0], conf[0][1]] = update_neighbours(neighbours[conf[0][0], conf[0][1]], neighbours[conf[1][0], conf[1][1]])
                counts[conf[1][0], conf[1][1]] = 0
    # Assign each field to the color with its coordinates, counts and neighbour counts
    for row in range(0, individual.shape[0]):
        for col in range(0, individual.shape[1]):
            if counts[row, col]:
                neighbours_num = dict()
                for color in neighbours[row, col]:
                    neighbours_num[color] = len(set(neighbours[row, col][color]))
                fields.setdefault(individual[row, col], []).append(((row, col) ,counts[row, col], neighbours_num))
    # Calculate the pixel counts of each color
    color_numbers = [0] * COLORS
    for color in fields:
        for field in fields[color]:
            color_numbers[color - 1] += field[1]
    print(counts)
    print(neighbours)
    print(individual)
    print(fields)
    print(color_numbers)


# def draw(individual):
#     for row in range(0, individual.shape[0]):
#         for col in range(0, individual.shape[1]):
#             color = individual[row, col]
#             if color == 1:
#                 individual[row, col] = [255,0,0]
#             elif color == 2:
#                 individual[row, col] = [0,255,0]
#             elif color == 3:
#                 individual[row, col] = [0,0,255]
#    plt.imshow(individual, interpolation="nearest")
#    plt.show()


img = toolbox.individual()
color_fields(img)
# draw(img)