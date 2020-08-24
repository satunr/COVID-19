import math
import matplotlib.pyplot as plt
import pickle

import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares


def instantiate():
    Duration = 121
    eB = 5
    iho = 12

    monitor = [12, 24, 48, 60, 72, 84, 108, 120]

    actual_inf = [5465, 3878, 1889, 873, 665, 376, 442, 432, 349]
    actual_inf = [float(v) / float(eB) for v in actual_inf]
    print (actual_inf)
    exit(1)

    # Density
    density = [13006, 13957, 27544, 8018, 3150]

    # Infection rate
    p = 0.01

    P = [1418207, 2559903, 1628706, 2253858, 476143]
    I = [actual_inf[0] for i in range(eB)]
    D = [0, 0, 0, 0, 0]

    return Duration, eB, iho, monitor, actual_inf, density, p, P, I, D

def opt(x, b, mode):

    pe = x[0]
    velocity = x[1]
    gamma = x[2]

    global Duration, P, sigma, alphas, iho, p, density, monitor, actual_inf

    dense = density[b]
    pop = P[b]

    z0 = [P[b] - (pe * P[b] + I[b] + D[b]), pe * P[b], I[b], 0, D[b]]
    V = [z0[2]]
    for t in range(iho, Duration):

        if t % iho == 0:
            beta = p * math.sqrt(2) * math.pi * velocity * (dense * math.pow(10, -6)) * 1

            z0[0] = z0[0] - (beta * z0[0] * z0[2]) / pop

            new_infected = sigma * z0[1]
            z0[1] = z0[1] + (beta * z0[0] * z0[2]) / pop - new_infected
            z0[2] = z0[2] + new_infected - gamma * z0[2]
            z0[3] = z0[3] + gamma * (1.0 - alphas) * z0[2]
            z0[4] = z0[4] + gamma * alphas * z0[2]

            if t in monitor:
                V.append(z0[2])

    a = np.asarray(actual_inf, dtype=np.float)
    b = np.asarray(V, dtype=np.float)

    if mode == 1:
        plt.plot([t * 15 for t in range(len(actual_inf))], actual_inf, marker = 'o', label = 'actual')
        plt.plot([t * 15 for t in range(len(actual_inf))], V, marker = 'o', label = 'predicted')

        plt.legend(fontsize = 15)
        plt.xlabel('Duration in days', fontsize = 15)
        plt.ylabel('Number of infected', fontsize = 15)

        plt.tight_layout()

        # plt.savefig('Fit.png', dpi = 300)
        # plt.show()

    if mode == 0:
        return np.sum(np.where(a != 0, a * np.log(a / b), 0))
    return V


Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

# SEIRD parameters
sigma, alphas = 0.25, 0.05
# pred_inf = opt(0.0001, 10.0, 0, 0.24)
#
# print (pred_inf)
#
# print (KLs(actual_inf, pred_inf))
# plt.plot([t * 15 for t in range(len(actual_inf))], actual_inf, marker = 'o', label = 'actual')
# plt.plot([t * 15 for t in range(len(actual_inf))], pred_inf, marker = 'o', label = 'predicted')
#
# plt.legend()
# plt.show()

x0 = [0.2, 100, 0.5]
res_1 = least_squares(opt, x0 = x0, bounds = ([0.0, 0.0, 0.0], [0.2, 1000.0, 0.5]), args = (3, 0))
x = res_1.x
print (x)
exit(1)

Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()
opt(x, 3, 1)
Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

V = []

V.append(opt([x[0], 1.0 * x[1], x[2]], 3, 1))
Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

V.append(opt([x[0], 2.5 * x[1], x[2]], 3, 1))
Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

V.append(opt([x[0], 4.0 * x[1], x[2]], 3, 1))
Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

V.append(opt([x[0], 6.5 * x[1], x[2]], 3, 1))
Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

V.append(opt([x[0], 8.0 * x[1], x[2]], 3, 1))
Duration, eB, iho, monitor, actual_inf, density, p, P, I, D = instantiate()

V.append(opt([x[0], 10.0 * x[1], x[2]], 3, 1))

print (V[0])

print ([pt[0] for pt in V])

X = [t * 15 for t in range(len(actual_inf))]

pickle.dump(V, open('V_u.p', 'wb'))
pickle.dump(X, open('X_u.p', 'wb'))
