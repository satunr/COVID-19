__author__ = "Satyaki Roy"
__version__ = "0"
__email__ = "satyakir@unc.edu"
__status__ = "Demo"
import random
import numpy as np
import pickle

from scipy.optimize import minimize
from scipy.spatial.distance import *

def squared_dist(a_x, a_y, b_x, b_y):
    return (a_x - b_x) ** 2 + (a_y - b_y) ** 2


def SimpleSortByXOptimization(X, S, dth):

    cnt = 0

    # points = [(X[i], X[i + len(S)], i) for i in range(len(S))]
    SEI_points = [(X[i], X[i + len(S)], i) for i in range(len(S)) if S[i] == 'S' or S[i] == 'E' or S[i] == 'I']
    SEI_points.sort()

    finish = 0
    for start in range(len(SEI_points)):
        c1 = [SEI_points[start][0], SEI_points[start][1]]

        while finish < len(SEI_points) and SEI_points[finish][0] - SEI_points[start][0] <= dth:
            finish += 1

        for index2 in range(start + 1, finish):
            c2 = [SEI_points[index2][0], SEI_points[index2][1]]

            if ((S[SEI_points[start][2]] == 'S' or S[SEI_points[start][2]] == 'E') and S[SEI_points[index2][2]] == 'I') or \
                    ((S[SEI_points[index2][2]] == 'S' or S[SEI_points[index2][2]] == 'E') and S[SEI_points[start][2]] == 'I'):
                if squared_dist(c1[0], c1[1], c2[0], c2[1]) < dth * dth:
                    cnt += 1.0

    return cnt


def SortByXOptimization(X, S, dth):

    cnt = 0

    S_points = [(X[i], X[i + len(S)], i) for i in range(len(S)) if S[i] == 'S']
    SEI_points = [(X[i], X[i + len(S)], i) for i in range(len(S)) if S[i] != 'R' and S[i] != 'D']

    n_ticks = 0

    for center in range(len(S_points)):
        c1 = [S_points[center][0], S_points[center][1]]

        possible_points = []

        for i in range(len(SEI_points)):
            if squared_dist(c1[0], c1[1], SEI_points[i][0], SEI_points[i][1]) < dth * dth and SEI_points[i][2] != S_points[center][2]:
                possible_points.append(SEI_points[i])


        for second_ind in range(len(possible_points)):
            for third_ind in range(second_ind + 1, len(possible_points)):
                n_ticks += 1
                if S[possible_points[second_ind][2]] != 'I' and S[possible_points[third_ind][2]] != 'I':
                    continue

                if squared_dist(possible_points[second_ind][0], possible_points[second_ind][1],
                             possible_points[third_ind][0], possible_points[third_ind][1]) >= dth * dth:
                    continue

                if S[possible_points[second_ind][2]] == 'S' and S[possible_points[third_ind][2]] == 'I':
                    cnt += 0.5
                    continue

                if S[possible_points[second_ind][2]] == 'I' and S[possible_points[third_ind][2]] == 'S':
                    cnt += 0.5
                    continue

                if S[possible_points[second_ind][2]] == 'I' and S[possible_points[third_ind][2]] == 'I':
                    cnt += 1.0
                    continue

                if S[possible_points[second_ind][2]] == 'E' and S[possible_points[third_ind][2]] == 'I':
                    cnt += 1.0
                    continue

                if S[possible_points[second_ind][2]] == 'I' and S[possible_points[third_ind][2]] == 'E':
                    cnt += 1.0
                    continue

    return cnt


def ApplyOptimization(users, opt_type):
    users_copy = users.copy()
    users_copy = users_copy.values()

    C = [user[0] for user in users_copy] + [user[1] for user in users_copy]
    S = [user[2] for user in users_copy]

    N = len(C)

    # Distance in feet that can cause contact 364000 feet == 1 degree (latitude, longitude)
    dth = 6.0 / 364000.0

    # Maximum distance a person is allowed to move
    cd = 6.0 / 364000.0

    # Define initialization of variable to be optimized in the format [x0, x1, ....], [y0, y1, ....]
    x0 = C.copy()

    # Constrain a person to stay within 'cd' meters of his current location; we can allow negative coordinates
    bnds = [(x0[i] - cd, x0[i] + cd) for i in range(int(len(x0) / 2))] + \
           [(x0[i] - cd, x0[i] + cd) for i in range(int(len(x0) / 2), len(x0))]

    active = 0
    for i in range(len(S)):
        active += int(S[i] != 'D' and S[i] != 'R')

    res = list()

    if opt_type == 1:
        res = minimize(SimpleSortByXOptimization, x0, (S, dth), method='SLSQP', options={'eps': 1.0, 'maxiter': active * 6}, bounds=bnds)
    elif opt_type == 2:
        res = minimize(SortByXOptimization, x0, (S, dth), method='SLSQP', options={'eps': 1.0, 'maxiter':active * 6}, bounds=bnds)
    x0 = res.x

    user_keys = [user_key for user_key in  users.keys()]

    for i in range(len(S)):
        users[user_keys[i]] = x0[i], x0[i + len(S)], users[user_keys[i]][2]

    return users


def random_move(users):
    global cd
    for u in users.keys():
        x = random.uniform(users[u][0] - cd, users[u][0] + cd)
        y = random.uniform(users[u][1] - cd, users[u][1] + cd)

        users[u] = [x, y, users[u][2]]

    return users


def dist(users, u, v, cd):
    c_u = [users[u][0], users[u][1]]
    c_v = [users[v][0], users[v][1]]

    if euclidean(c_u, c_v) <= cd:
        return True

    return False


def spread(users):
    global rho, gamma, lambdas, beta, cd

    for u in users.keys():
        if users[u][2] != 'S':
            continue

        for v in users.keys():
            if users[v][2] != 'I':
                continue
            if u == v:
                continue
            # print ('****')
            if dist(users, u, v, cd) and random.uniform(0, 1) < beta:
                users[u] = [users[u][0], users[u][1], 'I']
                break

    for u in users.keys():

        if users[u][2] == 'I':
            if random.uniform(0, 1) < gamma * lambdas:
                users[u] = [users[u][0], users[u][1], 'D']

        elif users[u][2] == 'I':
            if random.uniform(0, 1) < gamma * (1.0 - lambdas):
                users[u] = [users[u][0], users[u][1], 'R']

        elif users[u][2] == 'E':
            if random.uniform(0, 1) < rho:
                users[u] = [users[u][0], users[u][1], 'I']

    return users


# users = {'u1': [41.2069510, 72.1249581, 'R'],
#          'u2': [41.2069521, 72.1249584, 'I'],
#          'u3': [41.2069513, 72.1249579, 'S'],
#          'u4': [41.2069505, 72.1249590, 'S'],
#          'u5': [41.2069504, 72.1249582, 'S']}

# N = 100
# users = {}
#
# # Prob. of susceptible and infected
# sP = 0.7
# iP = 1.0 - sP
# for u in range(N):
#     e = np.random.choice(['S', 'I'], size = 1, p = [sP, iP])
#     users[u] = [random.uniform(41.2069, 41.2079), random.uniform(72.124951, 72.12505), e[0]]
#
# print (users)
# pickle.dump(users, open('users.p', 'wb'))
iterations = 5
iterate = 0

L = []
T = 100
cd = 6.0 / 364000.0

# Exposed to infected
rho = 0.25
gamma = 0.1
lambdas = 0.05
beta = 0.55

random_behavior = 0.5
while iterate < iterations:

    print('Iterate:', iterate)
    users = pickle.load(open('users.p', 'rb'))

    l = []
    t = 0
    while t < T:
        if random.uniform(0, 1) <= random_behavior:
            users = random_move(users)
        else:
            users = ApplyOptimization(users, 2)

        users = spread(users)
        t = t + 1

        cnt = len([u for u in users.keys() if (users[u][2] != 'S' and users[u][2] != 'E')])
        l.append(cnt)

    L.append(l)
    iterate = iterate + 1
    print ('***', len([u for u in users.keys() if (users[u][2] != 'S' and users[u][2] != 'E')]))
print (L)

pickle.dump(L, open('D-2-0.5.p', 'wb'))