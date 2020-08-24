import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
import ast
import statsmodels.stats.api as sms
import scipy.stats as st

# 1. Infection curve
# fig, ax1 = plt.subplots()
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Velocity', color = 'green')
# ax1.plot(x, yV, color = 'green')
#
# ax2 = ax1.twinx()
# ax2.set_ylabel('Reward', color = 'red')
# ax2.plot(x, yR, color = 'red')

# Y25 = pickle.load(open('yV-25.p', 'rb'))
# Y50 = pickle.load(open('yV-50.p', 'rb'))
# Y75 = pickle.load(open('yV-75.p', 'rb'))
# Y100 = pickle.load(open('yV-100.p', 'rb'))
#
# X = [i for i in range(len(Y100))]
#
# print (Y100)
#
# plt.plot(X, Y25, c = 'r', label = '25')
# plt.plot(X, Y50, c = 'g', label = '50')
# plt.plot(X, Y75, c = 'b', label = '75')
# plt.plot(X, Y100, c = 'black', label = '100')
#
# plt.legend()
# plt.tight_layout()  # otherwise the right y-label is slightly clipped
#
# plt.savefig('Figure-1.png')
# plt.show()

# 2. Correlation
# f = open('Res.txt', 'r')
# C = []
# for r in f.readlines():
#     x = ast.literal_eval(r[:-1])
#     C.extend(x)
#
# print (C)
# print (np.mean(C))
# print (np.std(C))
# print (st.t.interval(0.95, len(C)-1, loc=np.mean(C), scale=st.sem(C)))

# 3. Exponential decay

# V = [100.0, 250.0, 400.0, 650.0, 800.0, 1000.0]
# for v in V:
#     print (float(v)/(max(V)))
#     plt.plot([i * 0.1 for i in range(11)], [float(v)/(max(V)) * math.exp(-i * 0.1) for i in range(11)]
#              , marker = 'o', label = str(float(v)/1000.0) + ' km/h')
#
# plt.xlabel('Probability of queue', fontsize = 15)
# plt.ylabel('Reward', fontsize = 15)
# plt.legend(fontsize = 15)
#
# plt.tight_layout()
# plt.savefig('Exponential.png', dpi = 300)
# plt.show()

# 4. Correlation box plot
# construct some data like what you have:
x = np.random.randn(100, 8)
mins = np.array([0.38, 0.26, 0.15, 0.28, 0.19])
maxes = np.array([0.47, 0.37, 0.28, 0.37, 0.27])
means = np.array([0.43, 0.32, 0.21, 0.33, 0.23])

# create stacked errorbars:
plt.errorbar(np.arange(5), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw = 2, capsize = 10)
plt.xlim(-1, 5)

plt.xlabel('Borough', fontsize = 15)
plt.ylabel('Correlation coefficient', fontsize = 15)
plt.xticks((0, 1, 2, 3, 4), ('Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'Staten Island'))

plt.tight_layout()
plt.savefig('Correlate.png', dpi = 300)
plt.show()