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
            prbility = [m[j, n] / sum(m[:, n]) for j in range(len(Grid))]
            dest = np.random.choice([g for g in range(len(Grid))], p = prbility, size = 1)

        # For Random (uncomment)
        if mode == 0:
            dest = np.random.choice([g for g in range(len(Grid))],
                                    p = [1.0 / len(Grid) for _ in range(len(Grid))], size = 1)

        dest = dest[0]
        grid = Grid[dest]
        new_L[n] = (random.uniform(grid[0], grid[1]), random.uniform(grid[2], grid[3]))

    return new_L


def getLoc_Block(I, Grid):
    new_L = {}

    for n in range(N):
        grid_n = getGridIdx(L[n])
        block_id = 0

        for b, grids in Blocks.items():
            if grid_n in grids:
                block_id = b
                break

        # Probability distribution for the given list and the remaining numbers
        probabilities = [99/len(Blocks[block_id]) if i in Blocks[block_id] else 1/(len(Grid) - len(Blocks[block_id])) for i in range(len(Grid))]
        probabilities = probabilities / np.sum(probabilities)
        
        dest = np.random.choice([g for g in range(len(Grid))],
                                p = probabilities, size = 1)

        dest = dest[0]
        grid = Grid[dest]
        new_L[n] = (random.uniform(grid[0], grid[1]), random.uniform(grid[2], grid[3]))

        destination = getGridIdx(new_L[n])

        Mixing_Mat[grid_n, destination] += 1

    return new_L


def bounds(loc, X, Y):
    loc_updated = (np.clip(loc[0], 0, X), np.clip(loc[1], 0, Y))

    # if loc[0] != loc_updated[0] or loc[1] != loc_updated[1]:
    #     print ('***', loc, loc_updated)
    return loc_updated


def mean_confidence_interval(data, confidence, adjusted_SCP, adjust=0):
    global CI

    if adjust == 0:
        m = np.mean(data)
    else:
        m = adjusted_SCP
    n = len(data)
    h = CI[confidence] * np.std(data) / math.sqrt(n)

    return m, m - h, m + h


def sample(CP, sample_rate):
    global confidence, W
    sample_numbers = int(sample_rate * len(CP.keys()))

    Grid_indv = {i: [] for i in range(len(W))}
    for n in list(sorted(L.keys())):
        g = getGridIdx(L[n])
        Grid_indv[g].append(n)

    Grid_sample_nums = [int(s * sample_numbers) for s in W]

    test_cases = {i: [] for i in list(Grid_indv.keys())}
    for g in list(Grid_indv.keys()):
        print(len(Grid_indv), len(Grid_sample_nums), len(test_cases), g)
        test_cases[g] = list(random.sample(Grid_indv[g], Grid_sample_nums[g]))

    adjusted_SCP = 0
    denom_factor = 0
    for g in list(Grid_indv.keys()):
        denom_factor += ((1/(W[g])) * len([CP[u] for u in test_cases[g]]))
        adjusted_SCP += (1/(W[g])) * np.sum([CP[u] for u in test_cases[g]])

    adjusted_SCP /= denom_factor
    print(">>>> adjusted: ", adjusted_SCP)

    tests = []
    for k,v in test_cases.items():
        tests += list(v)
    data = []
    print("Total tests:", len(tests))
    for u in tests:
        data.append(CP[u])

    sm, low, high = mean_confidence_interval(data, confidence, adjusted_SCP, 0)
    adj_sm, adj_low, adj_high = mean_confidence_interval(data, confidence, adjusted_SCP, 1)

    real = np.mean(list(CP.values()))
    print (low, sm, real, high)
    print (adj_low, adj_sm, real, adj_high)

    if (real >= low and real <= high) and (real >= adj_low and real <= adj_high):
        return 1, ((low, high), real), 1, ((adj_low, adj_high), real)
    elif not (real >= low and real <= high) and (real >= adj_low and real <= adj_high):
        return 0, ((low, high), real), 1, ((adj_low, adj_high), real)
    elif (real >= low and real <= high) and not (real >= adj_low and real <= adj_high):
        return 1, ((low, high), real), 0, ((adj_low, adj_high), real)
    elif not (real >= low and real <= high) and not (real >= adj_low and real <= adj_high):
        return 0, ((low, high), real), 0, ((adj_low, adj_high), real)



def getGridIdx(Coords):
    global Grid

    idx = 0
    for g in Grid.values():
        if (g[0] < Coords[0] < g[1] and g[2] < Coords[1] < g[3]):
            break
        idx += 1

    return idx

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

def seird_var_R0(L, S, gamma, alpha, CPs, delta, nP = 8, dist=1.8288):
    global T, tested_infected, adj_tested_infected, isSuperSpreader, adj_conf_int, conf_int, D, a, model

    CPs1 = deepcopy(CPs)
    neighbors = {i: [] for i in L.keys()}

    l = list(sorted(L.keys()))
    S1 = np.zeros((len(l), len(l)))
    t0 = time.time()

    for t in range(24):
        # L = getLoc(L)
        
        L = getLoc_Block(I, Grid)

        each = int(len(l) / nP)

        with Pool(nP) as p:
            O = p.map(f, [[process, dist, L, l, each] for process in range(1, nP + 1)])

        for fs in O:
            S1 = S1 + fs

        S2 = deepcopy(S1)
        print (t, S2[np.where(S2 > 0)].size / 2, time.time() - t0)

    for person in l:
        neighbor = [j for j in l if S1[person, j] > 0]

        grid_number = getGridIdx(L[person])
        R0 = Gridwise_R0[grid_number]
        beta = gamma * R0
        r = 1.8288
        C = math.pi * math.pow(r, 2) * (float(N) / float(X * Y))
        prob = beta / C

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

                # Mixing_Mat[getGridIdx(L[person]), getGridIdx(L[other])] += 1

            if not flag:
                CPs1[person] = min(alpha_decay * CPs[person] + prob * sum([CPs[other] for other in neighbor]), 1.0)

    if T % 1 == 0:
        for r in range(len(sample_size)):
            t, interval, adj_t, adj_interval = sample(CPs1, sample_size[r])
            tested_infected[r].append(t)
            conf_int[r].append(interval)
            adj_tested_infected[r].append(adj_t)
            adj_conf_int[r].append(adj_interval)
    return S, CPs1


if __name__ == '__main__':  

    for iter in range(1):
        # initial status percentage lists
        E = [0.95, 0, 0.05, 0, 0]

        # Area in meters
        X, Y = 2000, 2000
        N = 5000

        # home percentage
        I = pickle.load(open("Home_Percentage"+str(N)+".p", "rb"))

        # Home address
        P = pickle.load(open("Home_freq"+str(N)+".p", "rb"))

        # Friendship matrix
        M = pickle.load(open("Friendship"+str(N)+".p", "rb"))

        # W = np.random.exponential(scale=1, size= 16)
        # W = [w/sum(W) for w in W]

        W = [0.3, 0.3] + [0.4 / (len(I) - 2) for i in range(len(I) - 2)]
        random.shuffle(W)

        W = [w/sum(W) for w in W]
        print(W, sum(W))

        pd = float(N) / (X * Y)
        status = ['S', 'E', 'I', 'R', 'D']
        sigma, gamma, alpha, delta = 0.25, 0.05, 0.05, 0.025
        alpha_decay = 1 / float(np.sqrt(10))

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
        Blocks = {0: [0, 1, 4, 5], 1: [2, 3, 6, 7], 2: [8, 9, 12, 13], 3: [10, 11, 14, 15]}
        Gridwise_R0 = [0 for j in range(len(Grid))]
        for k,v in Blocks.items():
            if k == 3:
                Left, Right = 1, 2.5
            elif k == 2:
                Left, Right = 2.5, 4.5
            elif k == 1:
                Left, Right = 4.5, 7
            elif k == 0:
                Left, Right = 7, 9.5
            
            for i in v:
                Gridwise_R0[i] = random.uniform(Left, Right)
        
        # Status of individuals
        temp_status = list("".join([(status[i] * int(E[i] * N)) for i in range(len(E))]))
        S = {i: temp_status[i] for i in range(N)}
        print (S)

        duration = 60
        T = 0

        sample_size = [0.2]
        tested_infected = [[] for i in range(len(sample_size))]
        conf_int = [[] for i in range(len(sample_size))]
        adj_tested_infected = [[] for i in range(len(sample_size))]
        adj_conf_int = [[] for i in range(len(sample_size))]
        isSuperSpreader = 0
        confidence = 95
        # random/superspreader = 0, latp = 1
        model = 0

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

        daily_indv_CPs = []
        Grid_Mean = {i: [] for i in range(len(W))}

        Mixing_Mat = np.zeros((len(Grid), len(Grid)))

        while T <= duration:

            print ('T: ', T)
            start = time.time()

            S, CPs = seird_var_R0(L, S, gamma, alpha, CPs, delta)
            T += 1

            print (len([u for u in S.keys() if S[u] == 'I']))
            
            # to calculate daily grid mean cp
            Grid_indv = {i: [] for i in range(len(W))}
            for n in list(sorted(L.keys())):
                g = getGridIdx(L[n])
                Grid_indv[g].append(n)
            
            for k,v in Grid_indv.items():
                Grid_Mean[k].append(np.mean([CPs[idx] for idx in v]))

            for i in range(len(sample_size)):
                print ("sample:", sample_size[i], np.sum(tested_infected[i]), np.sum(adj_tested_infected[i]), T)

            daily_indv_CPs.append(CPs)
            status_counts.append([list(S.values()).count(status[i]) for i in range(len(E))])
            Mean_CPs.append(np.mean(list(CPs.values())))
            print("Day:", T-1, status_counts[T-1], "CP:", Mean_CPs[T-1], "Time:", time.time() - start)

            pickle.dump([daily_indv_CPs, Grid_Mean, Mixing_Mat], open("daily_CP_for_dist_grid"+str(N)+"_adj.p", "wb"))


        # print(conf_int[0], len(conf_int[0]))
        # print(adj_conf_int[0], len(adj_conf_int[0]))
        # print(Mean_CPs, len(Mean_CPs))
        # print(CPs)
        print(Mixing_Mat)
        
        pickle.dump(Mean_CPs, open("Mean_CPs_Daily_grid"+str(N)+"_adj.p", "wb"))
        pickle.dump(status_counts, open("status_counts_grid"+str(N)+"_adj.p", "wb"))
        pickle.dump([conf_int[0], adj_conf_int[0]], open("sampling_inf_grid"+str(N)+"_adj.p", "wb"))
        
