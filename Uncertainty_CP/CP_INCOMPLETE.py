import random
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import time
import scipy.stats as st

from scipy.spatial.distance import *
from copy import deepcopy
from copy import deepcopy
from multiprocessing import Pool


def find_inflation(CP_sample, CPs1, S, inf_change, kappa = 0.001):

    global factor, Factor_Log, LogINC, LogCOM
    if len(CP_sample.keys()) == 0:
        return

    # factor = np.mean(list(CPs1.values())) - (np.mean(list(CP_sample.values())))
    factor = [1 - CP_sample[u] for u in inf_change if u in CP_sample.keys()]
    if len(factor) == 0:
        factor = 0
    else:
        factor = np.mean(factor)
    print ('***', factor)

    LogCOM.append(np.mean([CPs1[u] for u in CP_sample.keys()]))
    LogINC.append(np.mean([CP_sample[u] for u in CP_sample.keys()]))

    Factor_Log.append(factor)


def viz(Log):
    x = [each[1] for each in Log]
    y = [each[0] for each in Log]
    plt.scatter(x, y)
    plt.xlabel('Infected Status')
    plt.ylabel('Contagion Potential')
    plt.savefig('CPs.png')


def getLoc2(I, Grid, M, P, mode = 0):
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

    return loc_updated


def mean_confidence_interval(data, confidence = .95):
    global CI

    m = np.mean(data)

    # n = len(data)
    # h = CI[confidence] * np.std(data) / math.sqrt(n)

    int = st.t.interval(alpha = confidence,
                        df = len(data) - 1, loc = m, scale = st.sem(data))

    return m, int[0], int[1]


def sample(CP_sample, CP, sample_rate):

    global factor
    # sample_numbers = int(sample_rate * len(CP.keys()))
    # test_cases = random.sample(list(sorted(L.keys())), sample_numbers)

    test_cases = deepcopy(list(CP_sample.keys()))

    data = []
    for u in test_cases:
        data.append(CP[u])

    data = deepcopy([CP_sample[u] + factor for u in CP_sample.keys()])

    sm, low, high = mean_confidence_interval(data)
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


def seird(L, S, beta, prob, gamma, alpha, CPs, CP_sample, delta, sample_rate, nP = 8, dist=1.8288):
    global T, tested_infected, LogINC, LogCOM, isSuperSpreader

    CPs1 = deepcopy(CPs)

    l = list(sorted(L.keys()))
    S1 = np.zeros((len(l), len(l)))
    t0 = time.time()

    for t in range(4):
        # random/superspreader
        if model == 0:
            L = getLoc2(I, Grid, M, P, isSuperSpreader)
        # LATP
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

    print (LogINC)
    print (LogCOM, '\n')
    inf_change = []

    for person in l:
        neighbor = [j for j in l if S1[person, j] > 0]
        flag_sample, neighbor_sample = False, None
        if person in CP_sample.keys():
            flag_sample = True
            neighbor_sample = [j for j in neighbor if j in list(CP_sample.keys())]

        if S[person] == 'I':
            if random.uniform(0, 1) < gamma:
                S[person] = 'R'
                CPs1[person] = 0
                if flag_sample:
                    CP_sample[person] = 0

        elif S[person] == 'R':
            if random.uniform(0, 1) < delta:
                S[person] = 'S'

        elif S[person] == 'S':
            for other in neighbor:
                if random.uniform(0, 1) < beta * CPs[other]:
                    S[person] = 'I'
                    inf_change.append(person)
                    break

            CPs1[person] = alpha_decay * CPs[person] + prob * sum([CPs[other] for other in neighbor])
            CPs1[person] = min(CPs1[person], 1.0)

            if flag_sample:
                CP_sample[person] = min(alpha_decay * CP_sample[person]
                                        + prob * sum([CP_sample[other] for other in neighbor_sample]), 1.0)

    find_inflation(CP_sample, CPs1, S, inf_change)

    t, interval = sample(CP_sample, CPs1, sample_rate)
    tested_infected.append(t)
    conf_int.append(interval)
    
    return S, CPs1, CP_sample



if __name__ == '__main__': 

    for iter in range(13, 20):
        
        for rate in [0.1, 0.2, 0.3]:

            # initial status percentage lists
            E = [0.95, 0, 0.05, 0, 0]

            Log, LogINC, LogCOM, Factor_Log = [], [], [], []

            # Area in meters
            X, Y = 2000, 2000
            N = 5000

            # home percentage
            I = pickle.load(open("Home_Percentage5000.p", "rb"))

            # Home address
            P = pickle.load(open("Home_freq5000.p", "rb"))

            # Friendship matrix
            M = pickle.load(open("Friendship5000.p", "rb"))

            pd = float(N) / (X * Y)
            status = ['S', 'E', 'I', 'R', 'D']
            sigma, gamma, alpha, delta = 0.25, 0.05, 0.05, 0.025
            alpha_decay = 1 / float(np.sqrt(10))

            # Omicron R0 = 9.5; Delta: 3.2
            R0 = 3.2
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

            # Location of individuals
            # L = getLoc(None)
            L = getLoc2(I, Grid, M, P)

            # CPs of individuals
            CPs = {i: 1 if S[i] == 'I' else 0 for i in range(N)}
            for i in range(N):
                if S[i] == 'S':
                    CPs[i] = 0.001

            # For LATP model
            # Coordinates of destination
            D = {i: [(Grid[i][0] + Grid[i][1]) / 2, (Grid[i][2] + Grid[i][3]) / 2] for i in range(Grid_X * Grid_Y)}
            # Weighing factor (for LATP)
            a = 1.2

            sample_rate = rate
            sample_numbers = int(sample_rate * len(CPs.keys()))
            sample_numbers = random.sample(list(sorted(L.keys())), sample_numbers)
            CP_sample = {i: CPs[i] for i in sample_numbers}

            conf_int = []
            # random = 0, superSpreader = 1
            isSuperSpreader = 0
            confidence = 90
            # random/superspreader = 0, latp = 1
            model = 0

            tested_infected = []
            F = []
            for t in range(3):
                factor = pickle.load(open("pickle_files/factor_data/Inflate_Factor"+str(N)+"_"+str(t)+"_conf90_samp"+str(sample_rate*100)+".p", "rb"))
                F.append(np.mean(factor))
            factor = np.mean(F)
            print(">>>>", factor)
            # factor = 0
            
            status_counts = []
            Mean_CPs = []


            while T <= duration:

                print ('T: ', T, "sample: ", sample_rate, "conf:", confidence, "iter:", iter)
                start = time.time()

                S, CPs, CP_sample = seird(L, S, beta, prob, gamma, alpha, CPs, CP_sample, delta, sample_rate)
                T += 1
                print ("sample:", np.sum(tested_infected), T)

                status_counts.append([list(S.values()).count(status[i]) for i in range(len(E))])
                Mean_CPs.append(np.mean(list(CPs.values())))
                print("Day:", T-1, status_counts[T-1], "CP:", Mean_CPs[T-1], "Time:", time.time() - start)

            # viz(Log)

            pickle.dump(Mean_CPs, open("pickle_files/factor_data/Inflate_Mean_CPs"+str(N)+"_"+str(iter)+"_conf90_samp"+str(sample_rate*100)+".p", "wb"))
            pickle.dump(status_counts, open("pickle_files/factor_data/Inflate_status_counts"+str(N)+"_"+str(iter)+"_conf90_samp"+str(sample_rate*100)+".p", "wb"))
            pickle.dump(conf_int, open("pickle_files/factor_data/Inflate_sampling_N"+str(N)+"_"+str(iter)+"_conf90_samp"+str(sample_rate*100)+".p", "wb"))
            pickle.dump(Factor_Log, open("pickle_files/factor_data/Inflate_Factor"+str(N)+"_"+str(iter)+"_conf90_samp"+str(sample_rate*100)+".p", "wb"))


            plt.plot([i for i in range(len(LogINC))], [LogINC[i] for i in range(len(LogINC))], label = 'Incomplete')
            plt.plot([i for i in range(len(LogINC))], [LogCOM[i] for i in range(len(LogINC))], label = 'Complete')

            plt.xlabel('Time')
            plt.ylabel('CPs')
            plt.legend()

            plt.savefig('Factor.png')
            print("*****\n", Factor_Log)
