import pulp
import random
import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import operator
import itertools
import pandas as pd
import decimal
import time

from copy import *
from geopy import distance
from scipy import optimize
from sklearn.cluster import KMeans
from scipy.spatial.distance import *
from scipy import stats
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import silhouette_score, calinski_harabasz_score

np.printoptions(precision = 2)


def find_queue_probability(b_GDP, M, mus, mode):

    L = []
    for i in range(b_GDP):
        if mode == 0:
            lambdas = M[i, T]
        else:
            lambdas = np.mean(M[i, T - int(window): T])

        rho = float(lambdas) / float(mus)

        if rho <= 1:
            p0 = 1.0 - rho
            p1 = p0 * rho
            pq = 1.0 - (p0 + p1)

        else:
            pq = 1.0

        L.append(pq)

    mean_pq = np.mean(L)
    return mean_pq


def find_reward(velocity, mean_pq):

    global v, small_reward
    # return math.log(beta + small_reward)/(mean_pq + small_reward)
    # return (v.index(velocity) + 1) * math.exp(- int(math.floor(mean_pq * 10)))
    # return float(v.index(velocity) + 1)/float(len(v)) * math.exp(- mean_pq)
    return float(velocity)/(max(v)) * math.exp(- mean_pq)


class Node(object):

    def __init__(self, env, ID, velocity, dense, Z, pop, b_GDP, Q):
        global T, Duration

        self.ID = ID
        self.env = env
        self.z_total = Z

        self.velocity = velocity
        self.dense = dense
        self.pop = pop
        self.Q = Q

        # Number of ICUs proportional to GDP
        self.b_GDP = int(b_GDP)
        self.bed_queue = {j: [] for j in range(self.b_GDP)}

        self.last_state = (0, 0)
        self.current_state = (0, 0)
        self.next_state = None
        self.reward = 0

        self.beta = 1.0
        self.last_mean_pq = None

        # Arrival rates per bed
        self.M = np.zeros((self.b_GDP, Duration + 1))

        if self.ID == 1:
            self.env.process(self.time_increment())

        self.env.process(self.opt())
        self.env.process(self.inject_new_infection())
        self.env.process(self.treatment())
        # self.env.process(self.learn())

    def time_increment(self):

        global T, iho, epsilon, decay, window, x, yR, yV, v, Z_list, T_list, eB

        while True:

            T = T + 1
            # if T % iho == 0:
            #     print (T, epsilon)

            if T > 0 and T % window == 0:
                epsilon *= decay

            # if T % 10 == 0:
            #     Z_list.append([entities[b].z_total[2] for b in range(eB)])
            #     T_list.append(T)

            # if T > 0 and T % (100 * 24) == 0:
            #     epsilon = 0.6
            #     self.Q = [np.zeros((state_size, state_size)) for _ in range(len(capacity))]

            yield self.env.timeout(minimumWaitingTime)

    def treatment(self):

        global T, iho, epsilon, decay, window, recovery_time, hospital_recover

        while True:

            for i in range(self.b_GDP):
                if len(self.bed_queue[i]) == 0:
                    continue

                t = self.bed_queue[i][0]
                if t - T > recovery_time:
                    self.bed_queue[i].pop(0)
                    if random.uniform(0, 1) < hospital_recover:
                        self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] - 1,
                                        self.z_total[3] + 1, self.z_total[4]]
                    else:
                        self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] - 1,
                                        self.z_total[3], self.z_total[4] + 1]

            yield self.env.timeout(minimumWaitingTime)

    def inject_new_infection(self):

        global pI, cI

        while True:
            if T % fI == 0:
                n = float(cI) / float(T + 1) * 5
                self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] + n, self.z_total[3], self.z_total[4]]

            yield self.env.timeout(minimumWaitingTime)

    def opt(self):

        global iho, sigma, gamma, alphas, rho, p, pH, DHP
        while True:

            if T % iho == 0:
                z0 = deepcopy(self.z_total)

                self.beta = p * math.sqrt(2) * math.pi * self.velocity * (self.dense * math.pow(10, -6)) * 1

                z0[0] = z0[0] - (self.beta * z0[0] * z0[2]) / self.pop

                new_infected = sigma * z0[1]
                z0[1] = z0[1] + (self.beta * z0[0] * z0[2]) / self.pop - new_infected
                z0[2] = z0[2] + new_infected - gamma * z0[2]
                z0[3] = z0[3] + gamma * (1.0 - alphas) * z0[2]
                z0[4] = z0[4] + gamma * alphas * z0[2]

                self.z_total = deepcopy(z0)

                # Number of patients hospitalized
                nH = int(pH * new_infected)

                # Empty beds
                empty_beds = []
                for j in range(self.b_GDP):
                    if len(self.bed_queue[j]) == 0:
                        empty_beds.append(j)

                # Assign random hospital beds to patients if no beds are empty
                for i in range(nH):
                    if len(empty_beds) > 0:
                        bed = empty_beds.pop(0)
                    else:
                        bed = int(random.uniform(0, self.b_GDP - 1))

                    self.M[bed, T] += 1
                    self.bed_queue[bed].append(T)

                # Find mean probability of queue (pq)
                mean_pq = find_queue_probability(self.b_GDP, self.M, mus, 0)
                DHP[self.ID].append((nH, mean_pq, T))

                self.reward = find_reward(self.velocity, mean_pq)

                # Find capacity index
                cp = None
                for cp in range(len(capacity)):
                    if mean_pq >= capacity[cp]:
                        break

                self.Q[cp, v.index(self.velocity)] += self.reward

            yield self.env.timeout(minimumWaitingTime)

    def learn(self):
        global epsilon, state_size, lr, gamma_RL, mus, window, v, capacity, small_reward, yV, thr

        while True:
            if T > 10 and T % window == 0:

                mean_pq = find_queue_probability(self.b_GDP, self.M, mus, 1)

                if self.last_mean_pq is not None and abs(self.last_mean_pq - mean_pq) < thr:
                    self.last_mean_pq = mean_pq
                else:

                    # Find capacity index
                    cp = None
                    for cp in range(len(capacity)):
                        if mean_pq >= capacity[cp]:
                            break

                    if random.uniform(0, 1) < epsilon:
                        self.velocity = v[random.choice([i for i in range(state_size)])]
                    else:
                        self.velocity = v[np.argmax(self.Q[cp])]

                    if self.ID == 4:
                        yV.append(self.velocity)
                        yR.append(mean_pq)
                        x.append(T)

                    self.last_mean_pq = mean_pq

            yield self.env.timeout(minimumWaitingTime)


# Number of boroughs
eB = 5

# Interact how often in hours
iho = 12

# Simulation time in hours
Duration = 24 * 180

# Fraction of patients needing hospitalization
pH = 0.2

# SEIRD parameters
sigma, gamma, alphas = 0.25, 0.5, 0.05

# Minimum waiting time
minimumWaitingTime = 1

B = {0: 'Bronx', 1: 'Brooklyn', 2: 'Manhattan', 3: 'Queens', 4: 'Staten Island'}

# Real trends in infection in NYC
actual_inf = [5465, 3878, 1889, 873, 665, 376, 442, 432, 349]
actual_inf = [float(v)/float(eB) for v in actual_inf]

# Population
P = [1418207, 2559903, 1628706, 2253858, 476143]
# Infected
I = [50120, 63086, 30921, 68548, 14909]
# I = [actual_inf[0] for i in range(eB)]

# Death
# D = [4865, 7257, 3149, 7195, 1077]
D = [0, 0, 0, 0, 0]

# Beta parameters
# Density
density = [13006, 13957, 27544, 8018, 3150]
# Infection rate
p = 0.01

# Proportion of exposed
pe = 1.82911550e-04

# monitoring system for duration
small_reward = 0.00001
window = 10 * iho + iho/2

# Threshold to invoke RL
thr = 0

# Action space
# Velocity
v = [100.0, 250.0, 400.0, 650.0, 800.0, 1000.0]

# Queue probability less than
capacity = [0.66, 0.33, 0.0]
state_size = len(v)
indices = [(i, j) for i in range(len(capacity)) for j in range(len(v))]
# print (indices)
# exit(1)

# Percentage and count of new infections
pI = 0.0005
cI = 100000

# Frequency of new infections
fI = 24 * 30

# GDP parameter (in billion USD)
mid_B = 100
GDP = [42, 91, 600, 93, 14]
GDP = [((GDP[i] - np.mean(GDP)) / np.sum(GDP) + 1.0) * mid_B for i in range(eB)]

# queue model parameter
recovery_time = 14.0
hospital_recover = 0.8
mus = 1.0/float(recovery_time * 24)

iterate = 100

# Correlation list:
C_List = []

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Velocity', color='green')
#
# ax2 = ax1.twinx()
# ax2.set_ylabel('Probability of queue', color='red')

v_change = []
for iter in range(1):

    # Set the percent you want to explore (RL)
    epsilon = 0.75
    decay = 0.99
    lr = 0.3
    gamma_RL = 0.8

    Z_list, T_list = [], []
    print ('Iteration:', iter)

    # List for hospitalization and p(queue) correlation over time for each borough
    DHP = {b: [] for b in range(eB)}

    # Global time
    T = 0

    # Initial population
    Z = []
    for b in B.keys():
        zb = [P[b] - (pe * P[b] + I[b] + D[b]), pe * P[b], I[b], 0, D[b]]
        # print (zb, sum(zb))
        Z.append(zb)

    # Visualization
    yR = []
    yV = []
    x = []

    # Create SimPy environment and assign nodes to it.
    env = simpy.Environment()

    entities = [Node(env, i, v[0], density[i], Z[i], P[i], GDP[i], np.zeros((len(capacity), state_size))) for i in range(eB)]
    env.run(until = Duration)

    # print (yV)
    # # Number of time velocity changed
    # chn = 0
    # for j in range(1, len(yV)):
    #     if yV[j] != yV[j - 1]:
    #         chn += 1
    #
    # v_change.append(chn)
    # print(chn, np.mean(v_change), np.std(v_change))

    # print (x)
    # print (yR)

    # set of points with a dip in
    # declines in probability of queue
    # dec = []

    # Correlation between the increase in p(queue) and decrease in velocity
    # Cx = []
    # Cy = []
    #
    # for i in range(len(yR) - 1):
    #
    #     xv = v.index(yV[i])
    #     yv = yR[i]
    #
    #     Cx.append(-xv)
    #     Cy.append(yv)

    # curr = np.corrcoef(Cx, Cy)[0, 1]

    # if curr > 0.7:
    #     print (curr)
    #
    #     for pt in dec:
    #         plt.axvline(x = pt, alpha = 0.5)
    #
    #     for i in range(len(yR) - 1):
    #         if yR[i] > yR[i + 1]:
    #             ax1.scatter((i + 1) * window, v.index(yV[i]), c='green', s=4)
    #             ax2.scatter((i + 1) * window, yR[i], c='red', s=4)
    #
    #     ax1.plot(x, [v.index(pt) for pt in yV], color='green', alpha=0.5)
    #     ax2.plot(x, yR, color='red', alpha = 0.5)
    #     plt.grid()
    #     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #     plt.savefig('Inflection.png', dpi = 300)
    #     plt.show()
    #     break
    # pickle.dump(yV, open('yV-100.p', 'wb'))
    # C_List.append(np.corrcoef(Cx, Cy)[0, 1])
    # #
    # print (np.mean(C_List))
    # print (np.std(C_List))
    # print ('\n')

    # time.sleep(2)

    # print ('Infected')
    # print (np.mean([entities[u].z_total[2] + entities[u].z_total[3] + entities[u].z_total[4] for u in range(eB)]))
    # print (np.std([entities[u].z_total[2] + entities[u].z_total[3] + entities[u].z_total[4] for u in range(eB)]))
    #
    # print ('Death')
    # print (np.mean([entities[u].z_total[4] for u in range(eB)]))
    # print (np.std([entities[u].z_total[4] for u in range(eB)]))

    X = []
    L = DHP[0]
    X.append([pt[0] for pt in L])

    Hos_mean = [np.mean([DHP[b][t][0] for b in range(eB)]) for t in range(len(L))]
    Hos_std = [np.std([DHP[b][t][0] for b in range(eB)]) for t in range(len(L))]

    pqueue_mean = [np.mean([DHP[b][t][1] for b in range(eB)]) for t in range(len(L))]
    pqueue_std = [np.std([DHP[b][t][1] for b in range(eB)]) for t in range(len(L))]

    print (pqueue_mean)
    print (pqueue_std)

    pickle.dump(Hos_mean, open('Hos_mean.p', 'wb'))
    pickle.dump(Hos_std, open('Hos_std.p', 'wb'))
    pickle.dump(pqueue_mean, open('pqueue_mean.p', 'wb'))
    pickle.dump(pqueue_std, open('pqueue_std.p', 'wb'))

# print (C_List)
# print (np.mean(v_change), np.std(v_change))
plt.show()