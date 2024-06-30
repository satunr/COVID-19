import random
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
import pickle
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import *

# objective function
def g_opt(params, *args):
    
    gamma = params[0]
    
    gamma_dash = []
    for i in range(len(NR)):
        gamma_dash.append(float(NR[i]) / float(I_0[i]))

    squared_error = 0
    for i in range(len(NR)):
        squared_error += math.pow(gamma_dash[i] - gamma, 2)
        
    return squared_error

# ------------------- For Italy ** start ------------------------------

# --------- csv to data -----------
file = open("cov_data_italy_upd.csv", 'r')
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
# pickle.dump(dates, open("dates_ITA.p", "wb"))

# current I_0 at T = t
I_0 = []
for i in range(1, len(dates), 1):
    daily_data[dates[i]] = [cum_data[dates[i]][0] - cum_data[dates[i-1]][0], cum_data[dates[i]][1] - cum_data[dates[i-1]][1]]
    I_0.append(cum_data[dates[i]][0] - cum_data[dates[i]][1])
print(I_0)
# pickle.dump(daily_data, open("daily_data_ITA.p", "wb"))

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
# Contact Rate
C = math.pi * math.pow(r, 2) * pd
print("C:", C)
# param gamma
gamma = 0.01

params = [gamma, ]

# bounds for gamma
bnds = [(0, 0.05)]
    
print("------------------------")
result = minimize(g_opt, x0=params, method='SLSQP', bounds=bnds, options= {'disp':True})
print("opt gamma:", result.x)
