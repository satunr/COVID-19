import numpy as np
import random
import matplotlib.pyplot as plt

from copy import deepcopy


# Population
n0 = 1000
n = 2 * n0

N = [[i for i in range(n0)], [i for i in range(n0)]]
nContest = int(0.5 * n)

# Minimum population
K = 0.5

# Time
T = 50

# Payoff matrix for players 0 and 1
P0 = np.array([[1, 0], [3, 2]])
P1 = np.array([[3, 2], [1, 0]])

# Each player has a mixed strategy
# between Hawk (0) to Dove (1) and a fitness
D, Y_mn, Y_err = {0: {}, 1: {}}, [[], []], [[], []]

for i in range(n0):
    # Player of type 0 and 1 [p, f]
    # p: probability of playing dove
    # f: fitness of player
    D[0][i] = [random.uniform(0, 1), 0]
    D[1][i] = [random.uniform(0, 1), 0]

# Plotting (mean and standard deviation)
M0, M1 = [], []
S0, S1 = [], []

for t in range(T):
    print ('T. ', t)
    print ('Number of individuals. %d' % (n))

    # Two players compete many times
    for r in range(nContest):
        ind0 = random.choice(N[0])
        ind1 = random.choice(N[1])

        # Choose the strategies of players 0 and 1
        s0 = np.random.choice([0, 1], p = [D[0][ind0][0], 1 - D[0][ind0][0]], size = 1)[0]
        s1 = np.random.choice([0, 1], p = [D[1][ind1][0], 1 - D[1][ind1][0]], size = 1)[0]

        # Rewards of players 0 and 1
        r0 = P0[s0, s1]
        r1 = P1[s0, s1]

        # Update the fitness of players 0 and 1 by
        D[0][ind0][1] += r0
        D[1][ind1][1] += r1
    print (D)

    # Average fitness and strategy
    aS = [{i: D[0][i][0] for i in N[0]}, {i: D[1][i][0] for i in N[1]}]
    aF = [{i: D[0][i][1] for i in N[0]}, {i: D[1][i][1] for i in N[1]}]

    for i in [0, 1]:

        # Retain the fittest players
        best_players = sorted(aF[i].items(), key=lambda x: x[1], reverse = True)[: int(K * n0)]
        best_players = [ind for (ind, _) in best_players]
        D[i] = deepcopy({b: D[i][b] for b in best_players})

        # Create new individuals
        while len(D[i].keys()) < n0:
            # Replicate existing instance
            ind = random.choice(list(D[i].keys()))
            ind1 = max(list(D[i].keys())) + 1
            D[i][ind1] = [D[i][ind][0], 0]

        N[i] = list(D[i].keys())

    n = len(N[0]) + len(N[1])

    # Record Fitness
    M0.append(np.mean(list(aF[0].values())))
    M1.append(np.mean(list(aF[1].values())))
    S0.append(np.std(list(aF[0].values())))
    S1.append(np.median(list(aF[1].values())))

    # # Record Strategy
    # M0.append(np.mean(list(aS[0].values())))
    # M1.append(np.mean(list(aS[1].values())))
    # S0.append(np.std(list(aS[0].values())))
    # S1.append(np.median(list(aS[1].values())))

plt.plot([t for t in range(T)], np.array(M0), label = 'Player 0', color = 'green', linestyle = 'dashdot')
plt.fill_between([t for t in range(T)], np.array(M0) - np.array(M0),
                 np.array(M0) + np.array(M0), color = 'green', alpha = 0.1, linestyle = 'dotted')

plt.plot([t for t in range(T)], np.array(M1), label = 'Player 1', color = 'red', linestyle = 'dashdot')
plt.fill_between([t for t in range(T)], np.array(M1) - np.array(M1),
                 np.array(M1) + np.array(M1), color = 'red', alpha = 0.1)

plt.xlabel('Generations of evolutionary game', fontsize=15)
plt.ylabel('Median fitness', fontsize=15)
# plt.ylabel('Median mixed strategy [probability of Hawk]', fontsize=12)

plt.legend()
plt.tight_layout()
# plt.savefig('Strategy.png', dpi = 150)
plt.savefig('Fitness.png', dpi = 150)
plt.show()
