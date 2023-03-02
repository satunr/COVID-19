import random
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
import pickle
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import *
from scipy.special import gamma as gammaf
from numpy.linalg import norm, inv
# from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg

# ------------------------ for Italy *** start ------------------------
def Mu_R_opt(params, *args):
    
    epsilon = params[0]
    print("E:", epsilon)
    return epsilon
    
# --------- csv to data -----------
file = open("../cov_data_italy.csv", 'r')
R = file.readlines()

cum_data = {}
start_date = '12/31/21'
i = 0
print(len(R))
for k in R[1:]:
    k = k.strip().split(',')
    date = k[0]
    if date != start_date and len(cum_data) == 0:
        continue
    inf_count = k[6]
    recov_count = k[8]
    # print(k)
    if date not in cum_data:
        cum_data[date] = [0,0]
    
    cum_data[date][0] += int(inf_count)
    cum_data[date][1] += int(recov_count)

daily_data = {}
dates = list(cum_data.keys())

# current I_0 at T = t
I_0 = []
for i in range(1, len(dates), 1):
    daily_data[dates[i]] = [cum_data[dates[i]][0] - cum_data[dates[i-1]][0], cum_data[dates[i]][1] - cum_data[dates[i-1]][1]]
    I_0.append(cum_data[dates[i]][0] - cum_data[dates[i]][1])
print(I_0)

# ------------- end of csv to data -----------------------

# total population
N = 60000000 
# population density in /m^2
pd = 0.0002
# New Infected
NI = [v[0] / float(N) for k,v in daily_data.items()]
# New Recovered
NR = [v[1] / float(N) for k,v in daily_data.items()]
# current infected I_0
I_0 = [i / float(N) for i in I_0]
# contact sphere in m
r = 1.8288
# trip avg
trip_avg = 1
# Contact Rate
C = math.pi * math.pow(r, 2) * pd * trip_avg
print("C:", C)
# gamma
gamma = 0.04788463

#  param Beta = pd * Contact rate
beta = 0.5
# param mu
mu = [0.5 for i in range(len(NI))]
# param R_0
R_0 = [0.5 for i in range(len(NI))]
# epsilon
epsilon = 0.5

params = [epsilon] + [beta] + mu + R_0

# bounds for gamma, mu, and R_0
bnds = list(tuple((0, np.inf) for i in range(1)) + tuple((0.7*gamma, 1.2*gamma) for i in range(1)) + tuple((NR[i]/gamma, 1) for i in range(len(mu))) + tuple((0, 1 - NR[i]/gamma) for i in range(len(R_0))))
# constraints
cons = []
for i in range(len(NI)):
    def con(t, idx=i):
        epsilon = t[0]
        beta = t[1]
        mu = t[2:len(NI)+2]
        R_0 = t[len(NI)+2:]

        S_0 = [NI[i] / (mu[i] * beta) for i in range(len(mu))]
        I_0 = [NR[i] / gamma for i in range(len(NR))]

        return S_0[idx] + I_0[idx] + R_0[idx] + epsilon - 1
    cons.append({'type': 'eq', 'fun': con})
# ------------------------------------------------
# constraints
# this is the i-th element of cons(z):
def cons_i(params, i):
    epsilon = params[0]
    beta = params[1]
    mu = params[2: len(NI)+2]
    R_0 = params[len(NI)+2:]

    S_0 = [NI[j] / (mu[j] * beta) for j in range(len(mu))]
    I_0 = [NR[j] / gamma for j in range(len(NR))]

    return S_0[i] + I_0[i] + R_0[i] + epsilon - 1

# listable of scalar-output constraints input for SLSQP:
cons_per_i = [{'type':'eq', 'fun': cons_i, 'args': (i,)} for i in np.arange(len(NI))]
# ------------------------------------------------

print(len(mu), len(R_0), len(NI))

print("------------------------")
result = minimize(Mu_R_opt, x0=params, method='SLSQP', constraints=cons, bounds=bnds, options= {'disp':True})
print("------------------------")

# print(result.success, ":", result.message)
print(result)

res_epsilon = result.x[0]
res_beta = result.x[1]
res_mu = result.x[2: len(NI)+2]
res_R0 = result.x[len(NI)+2:]

print(np.mean(res_mu), np.std(res_mu))

res_S0 = [NI[i] / (res_mu[i] * res_beta) for i in range(len(res_mu))]
res_I0 = [NR[i] / gamma for i in range(len(NR))]


# ----------- timewise plot ------------------

x_pts = np.arange(len(res_I0))

fig, ax = plt.subplots()

ax.plot(x_pts, gaussian_filter1d(res_S0, sigma=2), label=r'$S_0$', color='green')
ax.plot(x_pts, gaussian_filter1d(res_I0, sigma=2), label=r'$I_0$', color='red')
ax.plot(x_pts, gaussian_filter1d(res_R0, sigma=2), label=r'$R_0$', color='blue')

ax.plot(x_pts, res_S0, color='green', linestyle="dashed", alpha=0.5)
ax.plot(x_pts, res_I0, color='red', linestyle="dashed", alpha=0.5)
ax.plot(x_pts, res_R0, color='blue', linestyle="dashed", alpha=0.5)
ax.plot(x_pts, [res_S0[i]+res_I0[i]+res_R0[i]+res_epsilon for i in range(len(res_I0))], label=r'$S_0 + I_0 + R_0$'  + '+ \u03B5' , linestyle='dotted', color='purple')
plt.axhline(y = N/N, color='orange', linestyle='dashed')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel("Timepoints", size=13)
ax.set_ylabel('SIRS', size=13)
# ax.set_ylim([100,190])

ax.legend(loc="center right")

fig.tight_layout()
plt.show()

# ------------------------ for Italy *** end ------------------------
