import random
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
import pickle
import math
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from multiprocessing import Pool


def test(S, test_rate = 0.2):
    test_numbers = int(test_rate * len(S.keys()))
    test_cases = random.sample(list(sorted(L.keys())), test_numbers)
    tested_inf = 0

    for u in test_cases:
        if S[u] == "I":
            tested_inf += 1

    return tested_inf


def getLoc(I, Grid, M, P, mode = 0):
    new_L = {}
    m = None

    # for superspreader
    if mode == 1:
        m = np.zeros((len(Grid), N))
        for i in range(len(Grid)):
            for j in range(N):
                numerator = sum([M[j, k] for k in range(N) if P[k] == i])
                denom = len([k for k in range(N) if P[k] == i])
                m[i, j] = float(numerator) / float(denom)

    for n in range(N):
        dest = None

        # For superspreader
        if mode == 1:
            prob = [m[j, n] / sum(m[:, n]) for j in range(len(Grid))]
            dest = np.random.choice([g for g in range(len(Grid))], p = prob, size = 1)

        # For Random
        if mode == 0:
            dest = np.random.choice([g for g in range(len(Grid))],
                                    p = [1.0 / len(Grid) for _ in range(len(Grid))], size = 1)
        
        dest = dest[0]
        grid = Grid[dest]
        new_L[n] = (random.uniform(grid[0], grid[1]), random.uniform(grid[2], grid[3]))

    return new_L


def f(input):
    [x, dist, L, l, each] = input
    Slocal = np.zeros((len(l), len(l)))

    # checking using euclidean distance if two person are neighbours
    for u in l[(x - 1) * int(each): x * int(each)]:
        for j in range(u + 1, len(l)):
            if euclidean(L[u], L[l[j]]) < dist:
                Slocal[u, l[j]] = 1
                Slocal[l[j], u] = 1
    return Slocal


def SIRS(L, S, beta, prob, gamma, alpha, CPs, delta, nP = 8, dist=1.8288):
    global T, tested_infected, mode

    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))
    S1 = np.zeros((len(l), len(l)))
    t0 = time.time()

    for t in range(24):
        L = getLoc(I, Grid, M, P, mode)

        each = int(len(l) / nP)

        with Pool(nP) as p:
            O = p.map(f, [[process, dist, L, l, each] for process in range(1, nP + 1)])

        for fs in O:
            S1 = S1 + fs

        S2 = deepcopy(S1)
        print (t, S2[np.where(S2 > 0)].size / 2, time.time() - t0)

    for person in l:
        # The neighbor list for each
        neighbor = [j for j in l if S1[person, j] > 0]

        # The SIRS spatial model
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
                CPs1[person] = min(alpha_decay * CPs[person] + prob * sum([CPs[other] for other in neighbor]), 1.0)

    tested_infected.append(test(S))
    return S, CPs1


if __name__ == '__main__': 
    # initial status percentage lists
    E = [0.95, 0, 0.05, 0, 0]

    # Area in meters
    X, Y = 1200, 1200
    # Total Population
    N = 2000

    # friendship matrix
    M = pickle.load(open("../Friendship2000.p", "rb"))

    # Grid Area
    Grid = {}
    Grid_X, Grid_Y = 4, 4
    dx, dy = X / Grid_X, Y / Grid_Y
    t = 0

    for i in range(Grid_X):
        for j in range(Grid_Y):
            Grid[t] = [i * dx, (i + 1) * dx, j * dy, (j + 1) * dy]
            t += 1

    print (Grid)

    # home percentage (the probability distribution for each grid)
    I = pickle.load(open("../Home_Percentage.p", "rb"))

    # Home address (the grid number for each person)
    P = pickle.load(open("../Home_freq" + str(N) + ".p", "rb"))

    pd = float(N) / (X * Y)
    # Only the S, I, R are used because SIRS model
    status = ['S', 'E', 'I', 'R', 'D']
    sigma, gamma, alpha, delta = 0.25, 0.05, 0.05, 0.025
    # the decay factor for infectivity
    alpha_decay = 1 / float(np.sqrt(10))

    # Omicron R0 = 9.5; Delta: 3.2
    R0 = 9.5
    beta = gamma * R0
    # radius of contact
    r = 1.8288
    # Contact rate
    C = math.pi * math.pow(r, 2) * (float(N) / float(X * Y))
    prob = beta / C

    # Status of individuals
    temp_status = list("".join([(status[i] * int(E[i] * N)) for i in range(len(E))]))
    S = {i: temp_status[i] for i in range(N)}
    print (S)

    duration = 1
    T = 0
    tested_infected = []

    # Location of individuals
    L = getLoc(I, Grid, M, P)

    # mode 0 -> random, 1 -> superspreader
    mode = 0

    status_counts = []
    Mean_CPs = []

    # CPs of individuals
    CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
    for i in range(N):
        if S[i] == 'S':
            CPs[i] = random.choice([0, 0.1])

    while T <= duration:
        print ('T: ', T)
        start = time.time()

        S, CPs = SIRS(L, S, beta, prob, gamma, alpha, CPs, delta)
        T += 1

        print (len([u for u in S.keys() if S[u] == 'I']))
        print (np.mean(tested_infected))


        status_counts.append([list(S.values()).count(status[i]) for i in range(len(E))])
        Mean_CPs.append(np.mean(list(CPs.values())))

        print("Day:", T, status_counts[T-1], "CP:", Mean_CPs[T-1], "Time:", time.time()-start)


    print("Mean_CPs", Mean_CPs)
    print("---------------------")
    print("status_counts", status_counts)
    print("---------------------")
    print("tested_inf", tested_infected)