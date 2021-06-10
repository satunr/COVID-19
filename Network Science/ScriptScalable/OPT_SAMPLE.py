import random, time
import math
import pickle
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


def EI(G, W):
    W = {u: round(W[u], 1) for u in list(G.nodes())}
    I_E = [e for e in list(G.edges()) if W[e[0]] == W[e[1]]]
    O_E = [e for e in list(G.edges()) if W[e[0]] != W[e[1]]]

    return float(len(O_E) - len(I_E)) / float(len(list(G.edges())))


def check_move(p0, p1):
    if p0[0] == p1[0] and p0[1] == p1[1]:
        return 0
    return 1


def check(X, C, cd, N):
    for i in range(N):
        if euclidean([X[i], X[i + N]], C[i]) > cd:
            return False
    return True


def sopt1(p0, i):
    global G
    G.remove_node(i)
    G.add_node(i)
    for j in range(N):
        if i == j:
            continue
        if euclidean(p0, [x0[j], x0[j + N]]) <= cT:
            G.add_edge(i, j)

    score = find_score1(G, S, i, 1)
    return score


def find_score1(G, S, i, mode):
    if mode == 1:
        score = len([e for e in list(G.edges()) if (e[0] == i or e[1] == i) and ((S[e[0]] == 'S' and S[e[1]] == 'I')
                    or (S[e[1]] == 'S' and S[e[0]] == 'I'))])
    else:
        score = len([e for e in list(G.edges()) if ((S[e[0]] == 'S' and S[e[1]] == 'I')
                    or (S[e[1]] == 'S' and S[e[0]] == 'I'))])

    return score


def sopt2(p0, i):
    global G
    G.remove_node(i)
    G.add_node(i)

    for j in range(N):
        if i == j:
            continue
        if euclidean(p0, [x0[j], x0[j + N]]) <= cT:
            G.add_edge(i, j)

    score = find_score2(G, N, S, i, 1)
    return score


def find_score2(G, N, S, i, mode):
    score = 0
    if mode == 1:
        for j in range(N):
            if j <= i:
                continue
            for k in range(N):
                if k <= j:
                    continue
                if G.has_edge(i, j) and G.has_edge(j, k) and G.has_edge(i, k):
                    sets = [S[i], S[j], S[k]]
                    if 'I' in sets and 'S' in sets:
                        score += 1
    else:
        for i in range(N):
            for j in range(N):
                if j <= i:
                    continue
                for k in range(N):
                    if k <= j:
                        continue
                    if G.has_edge(i, j) and G.has_edge(j, k) and G.has_edge(i, k):
                        sets = [S[i], S[j], S[k]]
                        if 'I' in sets and 'S' in sets:
                            score += 1

    return score


def sopt3(p0, i):

    global G, N, S
    G.remove_node(i)
    G.add_node(i)

    for j in range(N):
        if i == j:
            continue
        if euclidean(p0, [x0[j], x0[j + N]]) <= cT:
            G.add_edge(i, j)

    CP1 = deepcopy(CP0)
    CP = count_cp(G, S, deepcopy(CP1), 1)

    score = find_score3(G, CP, i, 1)
    return score


def find_score3(G, CP, i, mode):
    if mode == 1:
        score = 0
        for e in G.edges():
            if e[0] == i or e[1] == i:
                score += abs(CP[e[0]] - CP[e[1]])
    else:
        score = 0
        for e in G.edges():
            score += abs(CP[e[0]] - CP[e[1]])

    return score


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
X, Y = 50.0, 60.0

# Number of people
N = 75

# Maximum iteration
mi = 100000

# Iterations
iterate = 25

# Convergence
th = 0.3

# Coordinates
Clist = pickle.load(open('Clist.p', 'rb'))
# States
Stlist = pickle.load(open('Slist.p', 'rb'))

# ------------------
# Array of running time and score
Tlist_F = []
Slist_F = []
S0list_F = [],

for N in [15, 30, 45, 60, 75, 90]:
    # for N in [750]:

    X = int(2 * N / 3)
    print (N, X)

    # Array of running time and score
    Tlist = []
    Slist = []
    S0list = []

    Old = []
    New = []

    for iter in range(iterate):
        # print (N, iter)

        # Location
        C = {i: [random.uniform(0, X), random.uniform(0, Y)] for i in range(N)}

        # State of each node
        # state = ['S', 'E', 'I', 'R', 'D']
        # S = {i: state[random.randint(0, 2)] for i in range(N)}
        S = {i: np.random.choice(['S', 'I'], p=[0.7, 0.3])[0] for i in range(N)}

        # C = Clist[N][iter]
        # S = Stlist[N][iter]
        # print ('Iteration ', iter, len(C))

        x0 = [C[i][0] for i in range(N)] + [C[i][1] for i in range(N)]
        G = create_graph(x0)

        CP0 = {u: 0 for u in range(N)}

        # old_score = np.mean([find_score1(G, S, i, 2) for i in range(N)])
        # old_score = np.mean([find_score2(G, N, S, i, 2) for i in range(N)])
        CP = count_cp(G, S, deepcopy(CP0), 1)
        old_EI = EI(G, CP)
        Old.append(old_EI)

        old_score = np.mean([find_score3(G, CP, i, 2) for i in range(N)])

        S0list.append(old_score)

        t0 = time.time()

        while True:
            move = 0
            for i in range(N):
                p0 = [x0[i], x0[i + N]]

                # Constrain a person to stay within 'cd' meters of his current location;
                # we can allow negative coordinates
                bnds = [(p0[0] - cd, p0[0] + cd)] + [(p0[1] - cd, p0[1] + cd)]

                res = minimize(sopt1, p0, options={'eps': 1.0, 'maxiter': mi},
                               bounds=bnds, method='SLSQP', args=(i, ))

                x0[i] = deepcopy(res.x[0])
                x0[i + N] = deepcopy(res.x[1])

                move += check_move(p0, [x0[i], x0[i + N]])

            if float(move) / float(N) <= th:
                break

        if not check(x0, C, math.sqrt(2) * cd, N):
            print ('ERROR --> REPEAT')
            continue

        time_taken = time.time() - t0
        Tlist.append(time_taken)

        # new_score = np.mean([find_score1(G, S, i, 2) for i in range(N)])
        # new_score = np.mean([find_score2(G, N, S, i, 2) for i in range(N)])

        G = create_graph(x0)
        CP = count_cp(G, S, deepcopy(CP0), 1)
        new_score = np.mean([find_score3(G, CP, i, 2) for i in range(N)])
        new_EI = EI(G, CP)
        New.append(new_EI)

        Slist.append(new_score)
        # print (N, iter, old_EI, new_EI)

    print (np.mean(Old), np.std(Old))
    print (np.mean(New), np.std(New))

    # input('')
    # Tlist_F.append((np.mean(Tlist), np.std(Tlist)))
    # S0list_F.append((np.mean(S0list), np.std(S0list)))
    # Slist_F.append((np.mean(Slist), np.std(Slist)))
    #
    # print (Tlist_F[-1])
    # print (S0list_F[-1])
    # print (Slist_F[-1])

# print (Tlist_F)
# print (S0list_F)
# print (Slist_F)