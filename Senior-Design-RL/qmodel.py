import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


environment_columns = 0
environment_rows = 0


def bad_state(current_row_index, current_column_index):
    if env[current_row_index, current_column_index] == 0.:
        return False
    else:
        return True


def starting_location():
    # get a random row and column index
    current_row_index = 0
    current_column_index = 0
    # continue choosing random row and column indexes until a non-terminal state is identified
    # (i.e., until the chosen state is a 'white square').
    #while bad_state(current_row_index, current_column_index):
        #current_row_index = np.random.randint(environment_rows)
        #current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index


def next_action(current_row_index, current_column_index, epsilon1):
    if np.random.random() < epsilon1:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)


def next_location(current_row_index, current_column_index, action_index1):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index1] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index1] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index1] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index1] == 'left' and current_column_index > 0:
        new_column_index -= 1
    elif actions[action_index1] == 'center':
        new_row_index += 0
        new_column_index += 0
    return new_row_index, new_column_index


with open('cluster_centers.txt', 'r') as file:
    data = file.read().splitlines()
    result = {}
    # Loop through each line of the data
    for line in data:
        # Split the line before the colon
        name, coordinates = line.split(':')
        # Remove any leading or trailing spaces
        name = name.strip()
        coordinates = coordinates.strip()
        # Remove the square brackets and replace the spaces with commas
        coordinates = coordinates.replace('[', '(').replace(']', ')').replace(' ', ',')
        # Append the coordinates to the result dictionary
        if name not in result:
            result[name] = []
        result[name].append(coordinates)
    # Print the result
    print(result)

numbered_coordinates_dict = {}

for name, coordinates in result.items():
    numbered_coordinates_dict[name] = {}
    for i, coordinate in enumerate(coordinates):
        numbered_coordinates_dict[name][i + 1] = coordinate
    if len(coordinates) % 2 == 1:
        numbered_coordinates_dict[name][len(coordinates) + 1] = None

print(numbered_coordinates_dict)

for name, coordinates in numbered_coordinates_dict.items():
    num_coordinates = len(coordinates)
    if num_coordinates in [4, 6, 8]:
        if num_coordinates == 4:
            environment_rows = 2
            environment_columns = 2
        elif num_coordinates == 6:
            environment_rows = 3
            environment_columns = 2
        elif num_coordinates == 8:
            environment_rows = 4
            environment_columns = 2
        environment = [['' for i in range(environment_columns)] for j in range(environment_rows)]
        for i in range(num_coordinates):
            row = i // environment_columns
            col = i % environment_columns
            environment[row][col] = coordinates[i + 1]
        print(name + ':')
        for row in environment:
            print(row)
        print()

infected = np.zeros((environment_rows, environment_columns))
popDensity = np.zeros((environment_rows, environment_columns))

with open('infectedcases.txt', 'r') as file2:
    data = file2.read().splitlines()

    # Loop through each line of the data and split it into numbers
    for i, line in enumerate(data):
        numbers = line.split()

        # Loop through each number and add it to the array
        for j, number in enumerate(numbers):
            infected[i][j] = float(number)

    # Print the array
    print(infected)

with open('popDensity.txt', 'r') as file3:
    data = file3.read().splitlines()

    # Loop through each line of the data and split it into numbers
    for i, line in enumerate(data):
        numbers = line.split()

        # Loop through each number and add it to the array
        for j, number in enumerate(numbers):
            popDensity[i][j] = int(number)

    # Print the array
    print(popDensity)

q_values = np.zeros((environment_rows, environment_columns, 5))

print(q_values)

actions = ['up', 'right', 'down', 'left', 'center']

# matrix = [[0 for j in range(6)] for i in range(6)]
env = np.full((environment_rows, environment_columns), 0.)
env[1, 1] = -100.

for row in env:
    print(row)

epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9
weight = 0.5
Ia = 0


for name in result:
    print(name)
    for episode in range(100):
        row_index, column_index = starting_location()

        while not bad_state(row_index, column_index):
            action_index = next_action(row_index, column_index, epsilon)

            old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
            row_index, column_index = next_location(row_index, column_index, action_index)
            old_q_value = q_values[old_row_index, old_column_index, action_index]

        # Here implement the moving average
            Ia = (Ia * weight) + (infected[row_index, column_index] * (1 - weight))

        # New method for IA
            Ia = min(1.0, Ia)
            if popDensity[row_index, column_index] != 0 and abs(Ia - infected[row_index, column_index]) != 0:
                rewards = 1.0 / (popDensity[row_index, column_index] * abs(Ia - infected[row_index, column_index]))
            else:
                rewards = 0.0
            temporal_difference = rewards + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
    print(q_values)
    grid_rows = environment_rows
    grid_cols = environment_columns
    max_values = np.amax(q_values, axis=2)
    print(max_values)
    ax = sns.heatmap(max_values, linewidth=0.5, cmap='RdBu')
    plt.title(name, fontweight='bold', fontsize=16)
    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=14)
    plt.show()
