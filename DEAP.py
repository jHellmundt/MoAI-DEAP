%matplotlib inline
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
    color_numbers = [0] * COLORS
    counts = np.zeros(individual.shape, dtype=int) + 1
    neighbours = np.frompyfunc(dict, 0, 1)(np.empty((individual.shape[0], individual.shape[1]), dtype=object))
    # Fields: ('all field pixels', 'pixel count', 'all neighbour fields', ('max width', 'max height'))
    # 'all neighbour fields': dictionary with colors as key and list of ('field representative', 'number of neighbouring pixels')
    fields = dict()
    # Conflicts: dictionary with colors as key and list of ((x1, y1), (x2, y2))
    conflicts = dict()
    for i in range(1, COLORS + 1):
        conflicts[i] = []
        fields[i] = []

    # Count the number of pixels of the same color in each field
    for row in range(0, individual.shape[0]):
        for col in range(0, individual.shape[1]):
            color = individual[row, col]
            change = 0  # 0 = unchanged, 1 = vertical, 2 = horizontal

            # Get the neighbour above, if it's not of the same color
            if row - 1 >= 0 and color != individual[row - 1, col]:
                neighbours[row, col].setdefault(individual[row - 1, col], []).append((row - 1, col))

            # Get the neighbour to the right, if it's not of the same color
            if col - 1 >= 0 and color != individual[row, col - 1]:
                neighbours[row, col].setdefault(individual[row, col - 1], []).append((row, col - 1))

            # Get the neighbour from below
            if row + 1 < individual.shape[0]:
                # Set the change to vertical
                if color == individual[row + 1, col] and not change:
                    change = 1
                    # If there is a neighbour of the same color to the left: create a conflict between it
                    # and the neighbour of the same color below
                    if col + 1 < individual.shape[1] and color == individual[row + 1, col] and color == individual[
                        row, col + 1]:
                        conflicts[color] += [((row + 1, col), (row, col + 1))]
                # If there is no neighbour with the same color below, add the neighbour to the list
                else:
                    neighbours[row, col].setdefault(individual[row + 1, col], []).append((row + 1, col))

            # Get the neighbour to the left
            if col + 1 < individual.shape[1]:
                # Set the change to horizontal
                if color == individual[row, col + 1] and not change:
                    change = 2
                # If there is no neighbour with the same color to the left, add the neighbour to the list
                else:
                    neighbours[row, col].setdefault(individual[row, col + 1], []).append((row, col + 1))

            # Push current neighbours to the next pixel of the field if there will be a change
            # Change conflict field if the current field already had a conflict (Move conflict)
            if change == 1:
                counts[row + 1, col] += counts[row, col]
                counts[row, col] = 0
                neighbours[row + 1, col].setdefault(color, []).append((row, col))
                neighbours[row + 1, col] = update_neighbours(neighbours[row + 1, col], neighbours[row, col])

                conflicts[color] = [conf if conf[0] != (row, col) else ((row + 1, col), conf[1]) for conf in
                                    conflicts[color]]
                conflicts[color] = [conf if conf[1] != (row, col) else (conf[0], (row + 1, col)) for conf in
                                    conflicts[color]]
            elif change == 2:
                counts[row, col + 1] += counts[row, col]
                counts[row, col] = 0
                neighbours[row, col + 1].setdefault(color, []).append((row, col))
                neighbours[row, col + 1] = update_neighbours(neighbours[row, col + 1], neighbours[row, col])

                conflicts[color] = [conf if conf[0] != (row, col) else ((row, col + 1), conf[1]) for conf in
                                    conflicts[color]]
                conflicts[color] = [conf if conf[1] != (row, col) else (conf[0], (row, col + 1)) for conf in
                                    conflicts[color]]

    # Resolve conflicts
    for color in conflicts:
        for conf in conflicts[color]:
            if conf[0] != conf[1]:
                counts[conf[0][0], conf[0][1]] += counts[conf[1][0], conf[1][1]]
                neighbours[conf[1][0], conf[1][1]].setdefault(color, []).append((conf[1][0], conf[1][1]))
                neighbours[conf[0][0], conf[0][1]] = update_neighbours(neighbours[conf[0][0], conf[0][1]],
                                                                       neighbours[conf[1][0], conf[1][1]])
                counts[conf[1][0], conf[1][1]] = 0

    # Filter out all field representatives
    field_representatives = []
    for row in range(0, individual.shape[0]):
        for col in range(0, individual.shape[1]):
            if counts[row, col]:
                field_representatives.append((row, col))
                neighbours[row, col].setdefault(individual[row, col], []).append((row, col))

    # Assign each field to the color with its coordinates, counts, neighbour counts and its max. height and width
    for rep in field_representatives:
        neighbour_fields = dict()
        for color in neighbours[rep[0], rep[1]]:
            if color != individual[rep[0], rep[1]]:
                for o_rep in field_representatives:
                    if color in neighbours[o_rep[0], o_rep[1]] and individual[
                        o_rep[0], o_rep[1]] == color and o_rep != rep:
                        count = 0
                        for pixel in neighbours[o_rep[0], o_rep[1]][color]:
                            if pixel in neighbours[rep[0], rep[1]][color]:
                                count += 1
                        if count > 0:
                            neighbour_fields.setdefault(color, []).append((o_rep, count))
        # Get the maximum height and width (max_width/height = (smallest, biggest))
        max_width = (individual.shape[0], 0)
        max_height = (individual.shape[0], 0)
        for pixel in neighbours[rep[0], rep[1]][individual[rep[0], rep[1]]]:
            max_width = (min(max_width[0], pixel[0]), max(max_width[1], pixel[0]))
            max_height = (min(max_height[0], pixel[1]), max(max_height[1], pixel[1]))

        fields.setdefault(individual[rep[0], rep[1]], []).append((
                                                                 neighbours[rep[0], rep[1]][individual[rep[0], rep[1]]],
                                                                 counts[rep[0], rep[1]], neighbour_fields, (
                                                                 max_width[1] - max_width[0] + 1,
                                                                 max_height[1] - max_height[0] + 1)))

    # # The individual array (the picture)
    # print(individual)
    # print("")
    # # The counts of each color (index = color_num - 1)
    # print(color_numbers)
    # print("")
    # # Dictionary with all fields sorted by color as key and filled with list of fields
    # # Fields: ('all field pixels', 'pixel count', 'all neighbour fields', ('max width', 'max height'))
    # # 'all neighbour fields': dictionary with colors as key and list of ('field representative', 'number of neighbouring pixels')
    # for color in fields:
    #     print("***Color", color, ":")
    #     for field in fields[color]:
    #         print("Field pixels: ", field[0])
    #         print("Field pixel count: ", field[1])
    #         print("Maximal Width: ", field[3][0])
    #         print("Maximal Height: ", field[3][1])
    #         print("Neighbours:")
    #         for neigh_color in field[2]:
    #             print("  Color", neigh_color, ":")
    #             for neigh in field[2][neigh_color]:
    #                 print("    Representative:", neigh[0], ", number of neighbours from that field:", neigh[1])
    #         print("__________")
    #     print("")

    return fields


def color_count_eval(individual, fields):
    # Calculate the pixel counts of each color
    color_numbers = []
    for color in fields:
        for field in fields[color]:
            color_numbers[color - 1] += field[1]

    # Fitness Hyperparameters for color_counts (= cc)
    big_cc = COLORS / 4
    small_cc = COLORS / 4
    big_cc_border = individual.size - individual.size / 4
    small_cc_border = individual.size / 4
    big_cc_penalty = 1
    normal_cc_penalty = 1
    small_cc_penalty = 1
    big_cc_penalty_step_size = individual.size / 30
    normal_cc_penalty_step_size = individual.size / 30
    small_cc_penalty_step_size = individual.size / 30

    # Calculate Fitness Value for the color_counts (lower is better)
    # Function: (Distance to next border / Penalty Step Size) * Penalty for each step
    cc_fitness = 0
    counter_small = 0
    counter_normal = 0
    counter_big = 0
    for cc in color_numbers:
        if cc < small_cc_border:
            if counter_small > small_cc:
                cc_fitness += ((small_cc_border - cc) / small_cc_penalty_step_size) * small_cc_penalty
        elif cc > big_cc_border:
            if counter_big > big_cc:
                cc_fitness += ((cc - big_cc_border) / big_cc_penalty_step_size) * big_cc_penalty
        else:
            if counter_normal > COLORS - big_cc + small_cc:
                if counter_small > counter_big:
                    cc_fitness += ((big_cc_border - cc) / normal_cc_penalty_step_size) * normal_cc_penalty
                else:
                    cc_fitness += ((cc - small_cc_border) / normal_cc_penalty_step_size) * normal_cc_penalty

    return cc_fitness


ind = toolbox.individual()
fields_g = color_fields(ind)
cc_fitness_g = color_count_eval(ind, fields_g)

plt.imshow(ind, interpolation="nearest")
plt.show()