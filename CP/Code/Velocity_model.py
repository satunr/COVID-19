

import random
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
import pickle
import math

def move(L, pd, i, dist = 1.8288):
    global T
    # increase the possibility of interaction during certain period
    if T <= 20 or T > 40:
        v = 100
    else:
        v = 550

    # finding new location of each person
    new_L = {}
    for person in L.keys():
        r = v * 1
        theta = random.random() * 2 * math.pi
        new_point = (L[person][0] + math.cos(theta) * r, L[person][1] + math.sin(theta) * r)
        while ((0 > new_point[0] and new_point[0] > X) or (0 > new_point[1] and new_point[1] > Y)):
            theta = random.random() * 2 * math.pi
            new_point = (L[person][0] + math.cos(theta) * r, L[person][1] + math.sin(theta) * r)

        new_L[person] = new_point
    # print("->", i, "done moving")

    return new_L

def test(test_rate):
    test_numbers = int(test_rate * N)
    test_cases = random.sample(list(L.keys()), test_numbers)
    tested_inf = 0
    for i in test_cases:
        if S[i] == "I":
            tested_inf += 1

    return tested_inf


# SIRS model
def SIRS(L, S, beta, prob, gamma, alpha, CPs, delta, dist = 1.8288):
    global T
    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}
    for t in range(24):
        # new Location of each person
        L = move(L, pd, t)
        # finding the neighbors
        for person in sorted(L.keys()):
            neighbors[person] += [other for other in sorted(L.keys()) if euclidean(L[other], L[person]) < dist
                        and person != other]

    for person in sorted(L.keys()):
        neighbor = list(set(neighbors[person]))
        print(len(neighbor), "neihbo")

        # SIRS model
        if S[person] == 'I':
            if random.uniform(0, 1) < gamma:
                S[person] = 'R'
                CPs1[person] = 0

        elif S[person] == 'R':
            if random.uniform(0, 1) < delta:
                S[person] = 'S'

        elif S[person] == 'S':
            flag = False
            for other in neighbor:
                if random.uniform(0, 1) < beta * CPs[other]:
                    S[person] = 'I'
                    CPs1[person] = 1.0
                    flag = True
                    break

            if not flag:
                CPs1[person] = min(alpha_decay * CPs[person] +
                                  prob * sum([CPs[other] for other in neighbor]), 1.0)
            
    print("-> done SIRS", T)
    tested_infected.append(test(test_percent))
    tested_infected2.append(test(test_percent * 2))
    tested_infected3.append(test(test_percent * 4))
    tested_infected4.append(test(test_percent * 8))
    tested_infected5.append(test(test_percent * 12))
    tested_infected6.append(test(test_percent * 16))
    tested_infected7.append(test(test_percent * 20))
    tested_infected8.append(test(test_percent * 24))

    return S, CPs1

# ----------------- test for different initial seird percentage -----------------------
# initial status percentage lists
init_Zs = [[0.95, 0, 0.05, 0, 0]]

for Z in init_Zs:
    # Area in meters
    X, Y = 1750, 1750
    N = 5000
    pd = float(N) / (X * Y)
    # only S,I,R are used because of SIRS model
    status = ['S', 'E', 'I', 'R', 'D']
    sigma, gamma, alpha, delta = 0.25, 0.05, 0.05, 0.025
    alpha_decay = 1 / float(np.sqrt(10))
    beta = gamma * 4.5
    r = 1.8288
    C = math.pi * math.pow(r, 2) * (float(N) / float(X*Y))
    prob = beta / C

    # Location of individuals
    L = {i: (random.uniform(0, X), random.uniform(0, Y)) for i in range(N)}


    # Status of individuals
    temp_status = list("".join([(status[i] * int(Z[i] * N)) for i in range(len(Z))]))
    S = {i: temp_status[i] for i in range(N)}

    # CPs of individuals
    CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
    for i in range(N):
        if (S[i] == 'S'):
            CPs[i] = random.choice([0, 0.1])

    duration = 60
    T = 1
    status_counts = []
    Mean_CPs = []
    tested_infected = [0]
    tested_infected2 = [0]
    tested_infected3 = [0]
    tested_infected4 = [0]
    tested_infected5 = [0]
    tested_infected6 = [0]
    tested_infected7 = [0]
    tested_infected8 = [0]
    test_percent = 0.025

    Mean_CPs.append(np.mean(list(CPs.values())))
    status_counts.append([int(z * N) for z in Z])
    print("Day:", T-1, status_counts[T-1], "CP:", Mean_CPs[T-1])

    while (T <= duration):
        S, CPs = SIRS(L, S, beta, prob, gamma, alpha, CPs, delta)
        # print(list(S.values))
        status_counts.append([list(S.values()).count(status[i]) for i in range(len(Z))])
        Mean_CPs.append(np.mean(list(CPs.values())))
        # print("cp", CPs.values())

        print("Day:", T, status_counts[T-1], "CP:", Mean_CPs[T])

        T += 1
    
    print("----------------------\n")

print("Mean_CPs", Mean_CPs)
print("---------------------")
print("status_counts", status_counts)
print("---------------------")
print("tested_inf", tested_infected)