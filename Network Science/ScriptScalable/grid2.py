import random, time
import math, pickle
import networkx as nx
import numpy as np

from copy import deepcopy
from scipy.optimize import minimize
from multiprocessing import Pool
from scipy.spatial.distance import euclidean
np.printoptions(precision=2)


def check_move(p0, p1):
    if p0[0] == p1[0] and p0[1] == p1[1]:
        return 0
    return 1


def count_cp(G, S, CP, t, n):
    P = {u: 0 for u in G.nodes()}
    for u in P.keys():
        if S[n.index(u)] == 'I':
            P[u] = 1
        elif S[n.index(u)] == 'E' or S[n.index(u)] == 'S':
            P[u] = np.mean([P[v] for v in list(G.neighbors(u))])

    for u in CP.keys():
        if P[u] == 1:
            CP[u] = 1
        else:
            CP[u] = (CP[u] * t + P[u]) / (t + 1)

    return CP


def opt3(X, s, n):
    global N, S
    G = create_graph(X, n)
    CP1 = {u: 0 for u in n}
    CP = count_cp(G, s, deepcopy(CP1), 1, n)

    score = 0
    for e in G.edges():
        score += abs(CP[e[0]] - CP[e[1]])
    # print (score)
    return score


def create_graph2(X, n):

    global cT, N
    G = nx.Graph()
    G.add_nodes_from([i for i in n])
    for i in n:
        for j in n:
            if euclidean([X[n.index(i)], X[n.index(i) + len(n)]],
                         [X[n.index(j)], X[n.index(j) + len(n)]]) <= cT:
                G.add_edge(i, j)

    return G


def opt1(X, s, n):
    G = create_graph(X)
    score = len([e for e in list(G.edges()) if (s[n.index(e[0])] == 'S' and s[n.index(e[1])] == 'I')
                or (s[n.index(e[1])] == 'S' and s[n.index(e[0])] == 'I')])

    return score


def create_graph(X):

    global cT, N
    G = nx.Graph()
    G.add_nodes_from([i for i in range(N)])
    for i in range(N - 1):
        for j in range(i + 1, N):
            if euclidean([X[i], X[i + N]], [X[j], X[j + N]]) <= cT:
                G.add_edge(i, j)

    return G


def sopt1(p0, i, n, x0, s):

    G = create_graph2(x0, n)
    for j in n:
        if i == j:
            continue
        if euclidean(p0, [x0[n.index(j)], x0[n.index(j) + len(n)]]) <= cT:
            G.add_edge(i, j)

    score = find_score1(G, s, i, 1, n)
    return score


def find_score1(G, s, i, mode, n):
    if mode == 1:
        score = len([e for e in list(G.edges()) if (e[0] == i or e[1] == i) and
                     ((s[n.index(e[0])] == 'S' and s[n.index(e[1])] == 'I')
                    or (s[n.index(e[1])] == 'S' and s[n.index(e[0])] == 'I'))])
    else:
        score = len([e for e in list(G.edges()) if ((s[n.index(e[0])] == 'S' and s[n.index(e[1])] == 'I')
                    or (s[n.index(e[1])] == 'S' and s[n.index(e[0])] == 'I'))])

    return score


def f(x):
    global cd, bG

    s = [val[2] for val in x]
    n = [val[1] for val in x]
    x0 = [val[3][0] for val in x] + [val[3][1] for val in x]

    if len(x0) == 0:
        return []

    index = x[0][4]
    grids = x[0][0]

    xlow = grids[index][0][0]
    ylow = grids[index][0][1]
    xhigh = grids[index][1][0]
    yhigh = grids[index][1][1]
    # print ('*', index, xlow, xhigh, ylow, yhigh)

    # OPT APPROACH
    # Constrain a person to stay within 'cd' meters of his current location; we can allow negative coordinates (OPT)
    # bnds = [(max(xlow, x0[n.index(i)] - cd), min(xhigh, x0[n.index(i)] + cd)) for i in n] + \
    #        [(max(ylow, x0[n.index(i) + len(n)] - cd), min(yhigh, x0[n.index(i) + len(n)] + cd)) for i in n]
    #
    # # print ('**', bnds)
    #
    # res = minimize(opt1, x0, options={'eps': 1.0, 'maxiter': mi},
    #                bounds=bnds, method='SLSQP', args = (s, n,))

    # SAMPLING APPROACH
    while True:
        move = 0
        for i in n:
            p0 = [x0[n.index(i)], x0[n.index(i) + len(n)]]

            # Constrain a person to stay within 'cd' meters of his current location;
            # we can allow negative coordinates
            bnds = [(p0[0] - cd, p0[0] + cd)] + [(p0[1] - cd, p0[1] + cd)]

            res = minimize(sopt1, p0, options={'eps': 1.0, 'maxiter': mi},
                           bounds=bnds, method='SLSQP', args=(i, n, x0, s))

            x0[n.index(i)] = deepcopy(res.x[0])
            x0[n.index(i) + len(n)] = deepcopy(res.x[1])

            move += check_move(p0, [x0[n.index(i)], x0[n.index(i) + len(n)]])

        if float(move) / float(N) <= th:
            break

    return [n, x0, s]


def boundary_of_each_grid(B, size_of_each_grid):

    bG = {}
    grid_Id = 0

    #position of boundary grids
    bound = {}

    # Proceed row-wise
    number_of_rows = int(B[0][1] / size_of_each_grid[1][0])
    number_of_cols = int(B[1][1] / size_of_each_grid[1][1])
    print (number_of_rows, number_of_cols)

    for r in range(number_of_rows):

        for c in range(number_of_cols):

            bR = (r * size_of_each_grid[1][0], c * size_of_each_grid[1][0])

            bC = ((r + 1) * size_of_each_grid[1][1], (c + 1) * size_of_each_grid[1][1])

            bG[grid_Id] = [bR, bC]
            bound[grid_Id] = ''

            if r == 0:
                bound[grid_Id] = bound[grid_Id] + 'b'

            if r == number_of_rows - 1:
                bound[grid_Id] = bound[grid_Id] + 't'

            if c == 0:
                bound[grid_Id] = bound[grid_Id] + 'l'

            if c == number_of_cols - 1:
                bound[grid_Id] = bound[grid_Id] + 'r'

            grid_Id += 1

    return bG, bound


def define_grids(bG, bound, width):

    grid = {}
    for k in bG.keys():

        if bound[k] == 'bl':
            grid[k] = [(bG[k][0][0], bG[k][0][1]), (bG[k][1][0] + width, bG[k][1][1] + width)]

        elif bound[k] == 'b':
            grid[k] = [(bG[k][0][0], bG[k][0][1] - width), (bG[k][1][0] + width, bG[k][1][1] + width)]

        elif bound[k] == 'br':
            grid[k] = [(bG[k][0][0], bG[k][0][1] - width), (bG[k][1][0] + width, bG[k][1][1])]

        elif bound[k] == 'l':
            grid[k] = [(bG[k][0][0] - width, bG[k][0][1]), (bG[k][1][0] + width, bG[k][1][1] + width)]

        elif bound[k] == '':
            grid[k] = [(bG[k][0][0] - width, bG[k][0][1] - width), (bG[k][1][0] + width, bG[k][1][1] + width)]

        elif bound[k] == 'r':
            grid[k] = [(bG[k][0][0] - width, bG[k][0][1] - width), (bG[k][1][0] + width, bG[k][1][1])]

        elif bound[k] == 'tl':
            grid[k] = [(bG[k][0][0] - width, bG[k][0][1]), (bG[k][1][0], bG[k][1][1] + width)]

        elif bound[k] == 't':
            grid[k] = [(bG[k][0][0] - width, bG[k][0][1] - width), (bG[k][1][0], bG[k][1][1] + width)]

        elif bound[k] == 'tr':
            grid[k] = [(bG[k][0][0] - width, bG[k][0][1] - width), (bG[k][1][0], bG[k][1][1])]

    return grid


def create_grids(B, size_of_each_grid, width):

    bG, bound = boundary_of_each_grid(B, size_of_each_grid)

    grids = define_grids (bG, bound, width)
    return bG, grids


def check_pos(C, j, bG):
    global N

    xlow = bG[j][0][0]
    ylow = bG[j][0][1]
    xhigh = bG[j][1][0]
    yhigh = bG[j][1][1]
    # print (xlow, xhigh, ylow, yhigh)

    l = []
    for i in range(N):
        if C[i][0] >= xlow and C[i][0] <= xhigh and C[i][1] >= ylow and C[i][1] <= yhigh:
            l.append(i)

    return l


def opt2(X, s, n):
    G = create_graph(X, n)

    score = 0
    for i in n:
        for j in n:
            if j <= i:
                continue
            for k in n:
                if k <= j:
                    continue
                if G.has_edge(i, j) and G.has_edge(j, k) and G.has_edge(i, k):

                    sets = [s[n.index(i)], s[n.index(j)], s[n.index(k)]]
                    if 'I' in sets and 'S' in sets:
                        score += 1

    # print (score)
    return score


# Contact threshold (in feet)
cT = 6.0

# Distance threshold (in feet)
cd = 25.0

# Region
X, Y = 400.0, 400.0

# Number of people
N = 4000

# Maximum iteration
mi = 100000

# Iterations
iterate = 25

# Epidemic:
beta = 0.550
sigma = 0.25
gamma = 0.10
alpha = 0.05

# Convergence
th = 0.4

# Coordinates
Clist2 = pickle.load(open('Clist.p', 'rb'))
# States
Stlist2 = pickle.load(open('Slist.p', 'rb'))

B = [(0, X), (0, Y)]
size_of_each_grid = [(0, 0), (50.0, 50.0)]

Duration = 1

# Padding width
width = cd

bG, grids = create_grids(B, size_of_each_grid, width)
# grids = deepcopy(bG)

print (bG)
print (grids)

# Array of running time and score
Tlist_F = []
Slist_F = []
S0list_F = []
CM = {t: [] for t in range(Duration)}

for iter in range(iterate):
    time.sleep(10)
    print ('Iteration ', iter)
    p = Pool(processes = 16)


    # Location
    C = {i: [random.uniform(0, X), random.uniform(0, Y)] for i in range(N)}
    # C = Clist2[N][iter]

    # State of each node
    state = ['S', 'E', 'I', 'R', 'D']
    S = {i: state[random.randint(0, 2)] for i in range(N)}
    # S = Stlist2[N][iter]

    t0 = time.time()
    # print (C)

    for t in range(Duration):

        sc0 = opt1([C[i][0] for i in range(N)] + [C[i][1] for i in range(N)],
                  [S[i] for i in range(N)], [i for i in range(N)])

        L = p.map(f, [[[grids, i, S[i], C[i], j] for i in check_pos(C, j, bG)] for j in range(len(list(bG.keys())))])

        for j in list(bG.keys()):
            if len(L[j]) == 0:
                continue

            l = L[j]
            x0 = l[1]
            n = l[0]
            s = l[2]

            for i in n:
                C[i] = [x0[n.index(i)], x0[n.index(i) + len(n)]]
                S[i] = s[n.index(i)]

        for i in range(N):
            if S[i] == 'I':
                if random.uniform(0, 1) <= sigma * alpha:
                    S[i] = 'D'
                elif random.uniform(0, 1) <= sigma * (1.0 - alpha):
                        S[i] = 'R'
            elif S[i] == 'E':
                if random.uniform(0, 1) <= sigma:
                    S[i] = 'I'
            elif S[i] == 'S':
                for j in range(N):
                    if j <= i:
                        continue
                    if euclidean(C[i], C[j]) <= cT:
                        if S[j] == 'I' and random.uniform(0, 1) <= beta:
                            S[i] = 'E'
                            break

        # print (C)
        # sc = opt1([C[i][0] for i in range(N)] + [C[i][1] for i in range(N)],
        #             [S[i] for i in range(N)], [i for i in range(N)])
        cm = len([i for i in range(N) if S[i] == 'I' or S[i] == 'R' or S[i] == 'D'])
        print (t, cm)

        CM[t].append(cm)

    # print ([np.mean(CM[t]) for t in range(Duration)])
    # print ([np.std(CM[t]) for t in range(Duration)])

    tt = time.time() - t0
    print ('Time taken: ', tt)

    Tlist_F.append(tt)

    # Slist_F.append(sc)
    # S0list_F.append(sc0)
#
#     print(Tlist_F[-1])
#     print(Slist_F[-1])
#     print(S0list_F[-1])
#
    print(np.mean(Tlist_F), np.std(Tlist_F))
# print(np.mean(Slist_F), np.std(S0list_F))
# print(np.mean(S0list_F), np.std(S0list_F))
