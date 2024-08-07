import random
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
import pickle
import math
import matplotlib.pyplot as plt
import time
import scipy.stats
from copy import deepcopy
from multiprocessing import Pool


def getLoc2(I, Grid, M, P, mode):
    new_L = {}
    m = None

    if mode == 1:
        # matrix probability based on friendship and Home (Uncomment for superspreader)
        m = np.zeros((len(Grid), N))
        for i in range(len(Grid)):
            for j in range(N):
                numerator = sum([M[j, k] for k in range(N) if P[k] == i])
                denom = len([k for k in range(N) if P[k] == i])
                m[i, j] = float(numerator) / float(denom)

    for n in range(N):
        dest = None
        if mode == 1:
            # For superspreader (uncomment)
            prob = [m[j, n] / sum(m[:, n]) for j in range(len(Grid))]
            dest = np.random.choice([g for g in range(len(Grid))], p = prob, size = 1)

        # For Random (uncomment)
        if mode == 0:
            dest = np.random.choice([g for g in range(len(Grid))],
                                    p = [1.0 / len(Grid) for _ in range(len(Grid))], size = 1)

        dest = dest[0]
        grid = Grid[dest]
        new_L[n] = (random.uniform(grid[0], grid[1]), random.uniform(grid[2], grid[3]))

    return new_L


def bounds(loc, X, Y):
    loc_updated = (np.clip(loc[0], 0, X), np.clip(loc[1], 0, Y))

    # if loc[0] != loc_updated[0] or loc[1] != loc_updated[1]:
    #     print ('***', loc, loc_updated)
    return loc_updated


def mean_confidence_interval(data, confidence):
    global CI

    # a = 1.0 * np.array(data)
    # n = len(a)
    #
    # m, se = np.mean(a), scipy.stats.sem(a)
    # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    m = np.mean(data)
    n = len(data)
    h = CI[confidence] * np.std(data) / math.sqrt(n)

    return m, m - h, m + h


def sample(CP, sample_rate):
    global confidence
    sample_numbers = int(sample_rate * len(CP.keys()))
    test_cases = random.sample(list(sorted(L.keys())), sample_numbers)

    data = []
    for u in test_cases:
        data.append(CP[u])

    sm, low, high = mean_confidence_interval(data, confidence)
    real = np.mean(list(CP.values()))
    print (low, sm, real, high)

    if real >= low and real <= high:
        return 1, ((low, high), real)
    else:
        return 0, ((low, high), real)


def getLoc(L, dist = 100.0):
    global X, Y, N
    new_L = {}

    if L is None:
        for i in range(N):
            new_L[i] = (random.uniform(0, X), random.uniform(0, Y))
    else:
        for i in range(N):
            loc = (random.uniform(L[i][0] - dist, L[i][0] + dist),
                        random.uniform(L[i][1] - dist, L[i][1] + dist))
            new_L[i] = bounds(loc, X, Y)

    return new_L


def f(input):
    [x, dist, L, l, each] = input
    Slocal = np.zeros((len(l), len(l)))

    for u in l[(x - 1) * int(each): x * int(each)]:
        for j in range(u + 1, len(l)):
            if euclidean(L[u], L[l[j]]) < dist:
                Slocal[u, l[j]] = 1
                Slocal[l[j], u] = 1
    return Slocal


def latp(pt, D, a):

    # Available locations
    AL = [k for k in D.keys() if euclidean(D[k], pt) > 0]

    den = np.sum([1.0 / math.pow(float(euclidean(D[k], pt)), a) for k in sorted(AL)])

    plist = [(1.0 / math.pow(float(euclidean(D[k], pt)), a) / den) for k in sorted(AL)]

    next_stop = np.random.choice([k for k in sorted(AL)], p = plist, size = 1)

    # print (plist[next_stop[0]])

    return next_stop[0], D[next_stop[0]]

def seird(L, S, beta, prob, gamma, alpha, CPs, delta, nP = 8, dist=1.8288):
    global T, tested_infected, isSuperSpreader, conf_int, D, a, model

    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))
    S1 = np.zeros((len(l), len(l)))
    t0 = time.time()

    for t in range(24):
        # L = getLoc(L)
        
        # random/superspreader
        if model == 0:
            L = getLoc2(I, Grid, M, P, isSuperSpreader)
        # random/superspreader
        elif model == 1:
            for z in range(len(L)):
                L[z] = latp(L[z], D, a)[1]

        each = int(len(l) / nP)

        with Pool(nP) as p:
            O = p.map(f, [[process, dist, L, l, each] for process in range(1, nP + 1)])

        for fs in O:
            S1 = S1 + fs

        S2 = deepcopy(S1)
        print (t, S2[np.where(S2 > 0)].size / 2, time.time() - t0)

    for person in l:
        neighbor = [j for j in l if S1[person, j] > 0]

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

    for r in range(len(sample_size)):
        t, interval = sample(CPs1, sample_size[r])
        tested_infected[r].append(t)
        conf_int[r].append(interval)
    return S, CPs1


if __name__ == '__main__':  

    for iter in range(1, 5, 1):
        # initial status percentage lists
        E = [0.95, 0, 0.05, 0, 0]

        # Area in meters
        X, Y = 1250, 1250
        N = 2500

        # home percentage
        I = pickle.load(open("Home_Percentage"+str(N)+".p", "rb"))

        # Home address
        P = pickle.load(open("Home_freq"+str(N)+".p", "rb"))

        # Friendship matrix
        M = pickle.load(open("Friendship"+str(N)+".p", "rb"))

        pd = float(N) / (X * Y)
        status = ['S', 'E', 'I', 'R', 'D']
        sigma, gamma, alpha, delta = 0.25, 0.05, 0.05, 0.025
        alpha_decay = 1 / float(np.sqrt(10))

        # Omicron R0 = 9.5; Delta: 3.2; Alpha = 1.046
        R0 = 9.5
        beta = gamma * R0
        r = 1.8288
        C = math.pi * math.pow(r, 2) * (float(N) / float(X * Y))
        prob = beta / C

        CI = {90: 1.645, 95: 1.960, 99:	2.576}

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

        # Status of individuals
        temp_status = list("".join([(status[i] * int(E[i] * N)) for i in range(len(E))]))
        S = {i: temp_status[i] for i in range(N)}
        print (S)

        duration = 60
        T = 0

        sample_size = [0.1, 0.2, 0.3]
        tested_infected = [[] for i in range(len(sample_size))]
        conf_int = [[] for i in range(len(sample_size))]
        isSuperSpreader = 0
        confidence = 95
        # random/superspreader = 0, latp = 1
        model = 1

        status_counts = []
        Mean_CPs = []

        # Location of individuals
        # L = getLoc(None)
        L = getLoc2(I, Grid, M, P, isSuperSpreader)

        # CPs of individuals
        CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
        for i in range(N):
            if S[i] == 'S':
                CPs[i] = random.choice([0, 0.01])

        # For LATP model
        # Coordinates of destination
        D = {i: [(Grid[i][0] + Grid[i][1]) / 2, (Grid[i][2] + Grid[i][3]) / 2] for i in range(Grid_X * Grid_Y)}
        # Weighing factor (for LATP)
        a = 1.2

        while T <= duration:
            print ('T: ', T)
            start = time.time()

            S, CPs = seird(L, S, beta, prob, gamma, alpha, CPs, delta)
            T += 1

            print (len([u for u in S.keys() if S[u] == 'I']))
            

            for i in range(len(sample_size)):
                print ("sample:", sample_size[i], np.sum(tested_infected[i]), T)

            status_counts.append([list(S.values()).count(status[i]) for i in range(len(E))])
            Mean_CPs.append(np.mean(list(CPs.values())))
            print("Day:", T-1, status_counts[T-1], "CP:", Mean_CPs[T-1], "Time:", time.time() - start)


        pickle.dump(Mean_CPs, open("Mean_CPs"+str(N)+"_"+str(iter)+"_latp.p", "wb"))
        pickle.dump(status_counts, open("status_counts"+str(N)+"_"+str(iter)+"_latp.p", "wb"))
        pickle.dump(conf_int, open("sampling_N"+str(N)+"_"+str(iter)+"_latp.p", "wb"))
