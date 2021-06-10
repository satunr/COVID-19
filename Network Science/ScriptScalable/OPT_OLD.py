import random, time
import math, pickle
import networkx as nx
import numpy as np

from copy import deepcopy
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
np.printoptions(precision=2)


def create_graph(X):

    global cT, N
    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    for i in range(N - 1):
        for j in range(i + 1, N):
            if euclidean([X[i], X[i + N]], [X[j], X[j + N]]) <= cT:
                G.add_edge(i, j)

    return G


def opt1(X):
    G = create_graph(X)
    score = len([e for e in list(G.edges()) if (S[e[0]] == 'S' and S[e[1]] == 'I')
                or (S[e[1]] == 'S' and S[e[0]] == 'I')])
    # print (len(G.edges()), score)
    return score


def opt2(X):
    global N, S
    G = create_graph(X)

    score = 0
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                if G.has_edge(i, j) and G.has_edge(j, k) and G.has_edge(i, k):

                    sets = [S[i], S[j], S[k]]
                    if 'I' in sets and 'S' in sets:
                        score += 1

    # print (score)
    return score


def opt3(X):
    global N, S
    G = create_graph(X)
    CP1 = deepcopy(CP0)

    CP = count_cp(G, S, deepcopy(CP1), 1)

    score = 0
    for e in G.edges():
        score += abs(CP[e[0]] - CP[e[1]])
    # print (score)
    return score


def check(X, C, cd, N):
    for i in range(N):
        if euclidean([X[i], X[i + N]], C[i]) > cd:
            return False
    return True


def count_cp(G, S, CP, t):
    P = {u: 0 for u in G.nodes()}
    for u in P.keys():
        if S[u] == 'I':
            P[u] = 1
        elif S[u] == 'E' or S[u] == 'S':
            P[u] = np.mean([P[v] for v in list(G.neighbors(u))])

    for u in CP.keys():
        if P[u] == 1:
            CP[u] = 1
        else:
            CP[u] = (CP[u] * t + P[u]) / (t + 1)

    return CP


# Contact threshold (in feet)
cT = 6.0

# Distance threshold (in feet)
cd = 25.0

# Region
X, Y = 50.0, 50.0

# Number of people
N = 75

# Maximum iteration
mi = 100000

# Iterations
iterate = 25
# ------------------

# Array of running time and score
Tlist_F = []
Slist_F = []
S0list_F = []

# Coordinates
Clist = pickle.load(open('Clist.p', 'rb'))
# States
Stlist = pickle.load(open('Slist.p', 'rb'))

for N in [15, 30, 45, 60, 75, 90]:

    # Array of running time and score
    Tlist = []
    Slist = []
    S0list = []

    X = int(2 * N / 3)
    print (N, X)

    for iter in range(iterate):
        print ('Iteration ', iter)

        # Location
        # C = {i: [random.uniform(0, X), random.uniform(0, Y)] for i in range(N)}
        C = Clist[N][iter]

        # State of each node
        # state = ['S', 'E', 'I', 'R', 'D']
        # S = {i: state[random.randint(0, 2)] for i in range(N)}
        S = Stlist[N][iter]

        x0 = [C[i][0] for i in range(N)] + [C[i][1] for i in range(N)]

        # ------------------

        # # OPT1
        S0list.append(opt1(x0))

        # # OPT2
        # S0list.append(opt2(x0))

        # OPT3
        # CP0 = {u: 0 for u in range(N)}
        # S0list.append(opt3(x0))

        # ------------------

        # Constrain a person to stay within 'cd' meters of his current location; we can allow negative coordinates
        bnds = [(C[i][0] - cd, C[i][0] + cd) for i in range(N)] + [(C[i][1] - cd, C[i][1] + cd) for i in range(N)]

        t0 = time.time()
        res = minimize(opt1, x0, options={'eps': 1.0, 'maxiter': mi},
                       bounds=bnds, method='SLSQP')
        # print (res.x)
        # print (res.status, res.fun)
        # print (res.message)

        if not check(res.x, C, math.sqrt(2) * cd, N):
            print ('ERROR --> REPEAT')
            continue

        time_taken = time.time() - t0
        # print ('Time taken:', )
        Tlist.append(time_taken)
        Slist.append(res.fun)

    # print (np.mean(Tlist), np.std(Tlist))
    # print (np.mean(Slist), np.std(Slist))
    # print (np.mean(S0list), np.std(S0list))

    Tlist_F.append((np.mean(Tlist), np.std(Tlist)))
    Slist_F.append((np.mean(Slist), np.std(Slist)))
    S0list_F.append((np.mean(S0list), np.std(S0list)))

    print (Tlist_F[-1])
    print (Slist_F[-1])
    print (S0list_F[-1])

print (Tlist_F)
print (Slist_F)
print (S0list_F)

