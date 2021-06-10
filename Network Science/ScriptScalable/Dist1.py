import simpy
import pickle
import random
import numpy as np

from copy import deepcopy
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean


class Node(object):
    def __init__(self, env, ID, C):
        self.ID = ID
        self.C = C
        self.env = env
        self.nlist = []
        self.move_status = False

        self.env.process(self.time_increment())
        self.env.process(self.move())
        self.env.process(self.neighbor())

    def sopt1(self, X):
        global S
        score = 0
        for j in range(N):
            if i == j:
                continue
            if euclidean(X, entities[j].C) <= cT:
                L = [S[int(self.ID)], S[j]]
                if 'I' in L and 'S' in L:
                    score += 1

        return score

    def time_increment(self):
        global T, neighbor_sample, N, Log, S
        while True:
            if int(self.ID) == 1:
                T = T + 1
                # print (T, len([S[i] for i in range(N) if S[i] == 'I' or S[i] == 'R' or S[i] == 'D']))
                Log[T].append(len([S[i] for i in range(N) if S[i] == 'I' or S[i] == 'R' or S[i] == 'D']))

            if self.check_predecessors():
                self.move_status = True

            yield self.env.timeout(minimumWaitingTime)

    def check_predecessors(self):
        for u in self.nlist:
            if u == int(self.ID):
                continue
            if int(u) < int(self.ID) and entities[u].move_status is False:
                return False

        return True

    def move(self):
        global WAIT, N, X, Y, flout
        while True:
            if self.move_status:
                p0 = deepcopy(self.C)
                bnds = [(p0[0] - cd, p0[0] + cd)] + [(p0[1] - cd, p0[1] + cd)]

                r = random.uniform(0, 1)

                if r < flout:
                    self.C = [random.uniform(0, X), random.uniform(0, Y)]

                else:

                    res = minimize(self.sopt1, p0, options={'eps': 1.0, 'maxiter': mi},
                                   bounds=bnds, method='SLSQP')

                    if (res.x[0] < X and res.x[0] > 0) and (res.x[1] < Y and res.x[1] > 0):
                        self.C = deepcopy(res.x)

                self.move_status = False

                if int(self.ID) == N - 1:
                    self.change_status()
                    entities[0].move_status = True

            else:
                 WAIT[int(self.ID)] += 1

            yield self.env.timeout(minimumWaitingTime)

    def neighbor(self):
        global N, neighbor_sample, cT, T
        while True:
            self.nlist = []
            if T % neighbor_sample == 1:
                for u in range(N):
                    if int(u) == int(self.ID):
                        continue
                    if euclidean(entities[u].C, self.C) <= cT:
                        self.nlist.append(int(u))

            yield self.env.timeout(minimumWaitingTime)

    def change_status(self):
        global N, alpha, beta, gamma, sigma
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
                for j in entities[i].nlist:
                    if S[j] == 'I' and random.uniform(0, 1) <= beta:
                        S[i] = 'E'
                    break


# Contact threshold (in feet)
cT = 6.0

# Distance threshold (in feet)
cd = 25.0

# Region
Y = 50.0

# Number of people
N = 150

X = int(2 * N / 3)
# X = 100.0

# Maximum iteration
mi = 100000

# Iterations
iterate = 25

# Convergence
th = 0.5

# Epidemic:
beta = 0.550
sigma = 0.25
gamma = 0.10
alpha = 0.05

# Ignore optimizer
flout = 0.5

# Coordinates
Clist = pickle.load(open('Clist.p', 'rb'))
# print (Clist2[N])

Slist = pickle.load(open('Slist.p', 'rb'))

# Sampling interval
neighbor_sample = 3
minimumWaitingTime = 1

# Time
Duration = 100

Time = []
Log = {i: [] for i in range(1, Duration + 1)}

for iter in range(iterate):
    print ('Iteration:', iter)
    T = 0

    env = simpy.Environment()
    entities = []

    WAIT = {i: 0 for i in range(N)}

    S = Slist[N][iter]
    C0 = Clist[N][iter]
    # print (S)

    # C0 = {j: [random.uniform(0, X), random.uniform(0, Y)] for j in range(N)}
    # S = {i: np.random.choice(['S', 'I'], p=[0.7, 0.3])[0] for i in range(N)}

    for i in range(N):
        # entities.append(Node(env, str(i), Clist[N][iter][i]))
        entities.append(Node(env, str(i), C0[i]))

    env.run(until = Duration)
    Time.append(np.mean(list(WAIT.values())))
    print (Log)
    print (Time)

print (np.mean(Time))
print (np.std(Time))
print ([np.mean(Log[i]) for i in range(1, Duration + 1)])
print ([np.std(Log[i]) for i in range(1, Duration + 1)])

