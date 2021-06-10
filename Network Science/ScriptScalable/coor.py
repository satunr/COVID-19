import random
import pickle
import numpy as np


iterate = 50

# Region
Y, X = 50.0, 100.0
# X, Y = 100.0, 60.0

Clist = {}
Slist = {}

# for N in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
for N in [150]:

    X = int(2 * N / 3)
    print (N, X)

    Clist[N] = []
    Slist[N] = []

    for i in range(iterate):
        C = {j: [random.uniform(0, X), random.uniform(0, Y)] for j in range(N)}
        Clist[N].append(C)

        S = {i: np.random.choice(['S', 'I'], p = [0.7, 0.3])[0] for i in range(N)}
        print (S)
        Slist[N].append(S)

pickle.dump(Slist, open('Slist.p', 'wb'))
pickle.dump(Clist, open('Clist.p', 'wb'))


