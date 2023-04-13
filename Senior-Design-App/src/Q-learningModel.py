import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

environment_rows = 6
environment_columns = 6

q_values = np.zeros((environment_rows, environment_columns, 4))

actions = ['up', 'right', 'down', 'left']

rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 4] = 100.

aisles = {}
aisles[1] = [i for i in range(1, 5)]
aisles[2] = [1, 4]
aisles[3] = [i for i in range(1, 5)]
aisles[4] = [3]
aisles[5] = [2, 3]

for row_index in range(1, 6):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.

for row in rewards:
    print(row)


def bad_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True


def starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    while bad_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index


def next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)


def next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9
for episode in range(1000):
    row_index, column_index = starting_location()

    while not bad_state(row_index, column_index):
        action_index = next_action(row_index, column_index, epsilon)

        old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
        row_index, column_index = next_location(row_index, column_index, action_index)

        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print(q_values)


grid_rows = environment_rows
grid_cols = environment_columns

visual = np.zeros((grid_rows, grid_cols))

for i in range(grid_rows):
    for j in range(grid_cols):
        for k in range(4):
            if k == 0 and i != 0:
                visual[i-1, j] += q_values[i, j, k]
            elif k == 1 and j != environment_columns-1:
                visual[i, j+1] += q_values[i, j, k]
            elif k == 2 and i != environment_rows-1:
                visual[i+1, j] += q_values[i, j, k]
            elif k == 3 and j != 0:
                visual[i, j-1] += q_values[i, j, k]


ax = sns.heatmap(visual, linewidth=0.5, cmap='RdBu')

plt.title("Environment Heat Map")
plt.show()