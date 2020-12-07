#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pulp
import numpy as np
import os
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import *

from geopy import distance


def prospect(x, alpha, beta, lambdas):

    if x >= 0:
        return math.pow(x, alpha)

    return -lambdas * math.pow(-x, beta)


def count_extra(A):

    how_many_used = 0
    extra = 0

    for i in range(A.shape[0]):
        print (A[i])

        if np.sum(A[i]) > 0:
            how_many_used += 1

            if np.sum(A[i]) < 1.0:
                extra += 1.0 - np.sum(A[i])
    return how_many_used, extra

'''
def resource_allocation(T=100, p=0.3, fl=0.05, fh=0.2, Xlim=100, Ylim=100, z=10, warehouse=5,
                        filename='covid_confirmed_NY_updated'):
    # CSV file upload
    file = pd.read_csv("covid_confirmed_NY_updated.csv")
    file = file.iloc[0:z, :]

    # Number of warehouses
    how_many = warehouse

    # Coordinate of each zone --> C = {i: (random.uniform(0, Xlim), random.uniform(0, Ylim)) for i in range(z)}
    C = {}
    coordinates = np.array(file['Location'].values)
    for a in range(0, z):
        arr = coordinates[a].split(', ')
        for b in range(0, len(arr)):
            arr[b] = float(arr[b])
        C[a] = (arr[0], arr[1])

    # List of warehouses
    # LW = np.random.choice(list(C.keys()), size = how_many, replace = False)
    # LW = np.array([z - i for i in range(how_many)])
    LW = np.array([5, 6, 7, 8, 9])

    warehouses = []
    for i in LW:
        warehouses.append(C[i])

    array = np.zeros((z, len(LW)))
    for row in range(0, z):
        for column in range(0, len(LW)):
            # array[row][column] = distance.distance(C[row], warehouses[column]).miles
            array[row][column] = euclidean(C[row], warehouses[column])

    print('array:', array)

    avg_distance = (np.mean(array, axis=1)).reshape(z, )
    std = (np.std(array, axis=1)).reshape(z, )

    # Assign vaccine to warehouse
    # VW = {i: random.choice(LW) for i in range(T)}

    VW = {}
    current_warehouse = 0
    for f in range(1, T + 1):
        VW[f - 1] = LW[current_warehouse]
        if f % (T / warehouse) == 0:
            current_warehouse += 1

    # r = [random.uniform(0.1, 0.8) for i in range(z)]           # potency of the vaccine
    r = [0.2 for i in range(z)]  # potency of the vaccine
    B = np.array(file['Population Density'].values) * p  # rate of disease spread
    N = np.array(file['Population'].values)  # total population for each zone
    I = np.array(file['Total Infected'].values)  # total number infected in each zone
    S = N - I  # susceptible in each zone

    # Instantiate our problem class
    model = pulp.LpProblem("Vaccine problem", pulp.LpMinimize)

    X = pulp.LpVariable.dicts("X", ((i, j) for i in range(T) for j in range(len(B))), lowBound=0, upBound=1.0,
                              cat='Continuous')
    # Objective
    # model += np.sum([(B[b] * (S[b] - r[b] * pulp.lpSum([X[(j, b)] for j in range(T)])))
    # for b in range(z)])  # infected

    # den = float(T * np.max(array))
    # # den = 1.0
    # model += np.sum([X[j, b] * euclidean(C[VW[j]], C[b]) for j in range(T) for b in range(z)])/den  # economic

    # ---------------------------------------------------------------------------------------------------------
    # Objective functions

    den_infected = float(np.max(S) * np.max(B)*z)
    model += np.sum([(B[b] * (I[b] + 0.001) / (N[b] + 0.001) * (S[b] - r[b] * pulp.lpSum([X[(j, b)]
    for j in range(T)]))) for b in range(z)])/den_infected  # infected

    den_economic = float(T * np.max(array))
    model += np.sum([X[j, b] * euclidean(C[VW[j]], C[b])
    for j i n range(T) for b in range(z)])/den_economic  # economic

    # den_3 = float(np.max(I) * z)
    # model += np.sum((I[b] - (r[b] * pulp.lpSum([X[(j, b)] for j in range(T)]))) for b in range(z)) / den_3

    # Constraint 1
    for i in range(T):
        model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) == 1

    # Constraint 2
    s = 0.0
    for i in range(T):
        s += pulp.lpSum([X[(i, j)] for j in range(len(B))])

    model += s <= T

    # # Constraint 3 (fairness lower)
    # for i in range(len(B)):
    #     # model += pulp.lpSum([X[(j, i)] for j in range(T)]) >= min(S[i], fl * T) # vaccine
    #     model += pulp.lpSum([X[(j, i)] for j in range(T)]) >= min(I[i], fl * T) # drug

    # Constraint 4 (fairness upper)
    for i in range(len(B)):
        model += pulp.lpSum([X[(j, i)] for j in range(T)]) <= min(S[i], fh * T) # vaccine
        # model += pulp.lpSum([X[(j, i)] for j in range(T)]) <= min(I[i], fh * T) # drug

    # ---------------------------------------------------------------------------------------------------------

    # Constraints
    # model += 3 * A + 4 * B <= 30
    # model += 5 * A + 6 * B <= 60
    # model += 1.5 * A + 3 * B <= 21

    # # Constraint 1
    # for i in range(T):
    #     model += pulp.lpSum([X[(i, j)] for j in range(len(B))]) == 1
    #
    # # Constraint 2
    # s = 0.0
    # for i in range(T):
    #     s += pulp.lpSum([X[(i, j)] for j in range(len(B))])
    #
    # model += s == T
    #
    # # Constraint 3 (fairness upper)
    # for i in range(len(B)):
    #     model += pulp.lpSum([X[(j, i)] for j in range(T)]) >= fl * T
    #
    # # Constraint 4 (fairness lower)
    # for i in range(len(B)):
    #     model += pulp.lpSum([X[(j, i)] for j in range(T)]) <= fh * T
    #
    # # model += np.sum(cp.sum(x, axis = 0) == T)
    # # model += np.sum(x, axis = 0) >= int(f * T)

    model.solve()
    print(pulp.LpStatus[model.status])
    print (pulp.value(model.objective))
    # input('')

    # Transferred the pulp decision to the numpy array (A)
    A = np.zeros((T, len(B)))

    for i in range(T):
        for j in range(len(B)):
            A[i, j] = X[(i, j)].varValue

    # print (A)

    # Check total-sum:
    print (np.sum(A))
    input('')

    # Check row-sum:
    for i in range(T):
        print (np.sum(A[i]))
    input('')

    # print (A[:,8:10])

    # Check col-sum:

    # vaccines_per_zone = []
    # for i in range(len(B)):
    #     vaccines_per_zone.append(np.sum(A[:, i]))
    #
    # x_labels = []
    # for n in range(z):
    #     label_string = ''
    #     label_string += str(int(B[n])) + ', ' + "{:.5f}".format(I[n] / N[n]) + ", " + str(
    #         int(avg_distance[n])) + ", " + str(int(std[n]))
    #     x_labels.append(label_string)
    #
    # plt.figure(figsize=(25, 3))
    # plt.bar(x_labels, vaccines_per_zone, align='center', alpha=0.5)
    # plt.ylabel('No. of Vaccines')
    # plt.title('Vaccine Optimization → ONLY ECONOMIC Factor → ' + str(sorted(LW)))
    # plt.show()

    # plt.savefig('economic.png')
    # print(vaccines_per_zone)

os.chdir('/Users/satyakiroy/PythonCodes/Covid/Vaccine')
resource_allocation()
'''
# A = np.array([[0, 0, 1], [0, 0, 0.5], [0, 0, 0.25], [0, 0, 0]])
# how_many_used, extra = count_extra(A)
# print (how_many_used, extra)
# for i in range(-10, 10):
#     print (prospect(i, 0.5, 0.5, 1.5))

plt.plot([i for i in range(-50, 50)], [prospect(i, 0.2, 0.2, 2) for i in range(-50, 50)], linewidth = 3, label = 'lambda = 2.0', c = 'red')
plt.plot([i for i in range(-50, 50)], [prospect(i, 0.2, 0.2, 1) for i in range(-50, 50)], linewidth = 3, label = 'lambda = 1.0', c = 'green')
plt.plot([i for i in range(-50, 50)], [prospect(i, 0.2, 0.2, 0.5) for i in range(-50, 50)], linewidth = 3, label = 'lambda = 0.5', c = 'blue')

plt.grid()
plt.xlabel('Gain', fontsize = 15)
plt.ylabel('Utility', fontsize = 15)

plt.tight_layout()
plt.legend()
plt.savefig('Prospect.png', dpi = 300)
plt.show()