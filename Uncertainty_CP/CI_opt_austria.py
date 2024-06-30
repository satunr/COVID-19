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

'''
germany - 6/22
italy - 11/22

poland - 6/22
albania - 6/22
malta - 6/22
lithuania - 6/22
Austria - 6/22

czech - 12/21
hungary - 3/22
moldova - 4/22
denmark - 11/21
slovakia - 2/21
switzerland - 3/22
bosnia - 8/21
'''

# ------------------------ for Italy *** start ------------------------
def Mu_R_opt(params, *args):
    
    epsilon = params[0]
    print("E:", epsilon)
    return epsilon
    

# --------- csv to data -----------
file = open("../cov_data_austria_upd.csv", 'r', encoding='latin-1')
R = file.readlines()

cum_data = {}
start_date = '12/31/21'
i = 0
print(len(R))
for k in R[1:]:
    print(k)
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
pickle.dump(dates, open("../dates_AUS.p", "wb"))

# current I_0 at T = t
I_0 = []
for i in range(1, len(dates), 1):
    daily_data[dates[i]] = [cum_data[dates[i]][0] - cum_data[dates[i-1]][0], cum_data[dates[i]][1] - cum_data[dates[i-1]][1]]
    I_0.append(cum_data[dates[i]][0] - cum_data[dates[i]][1])
print(I_0)


pickle.dump(daily_data, open("../daily_data_AUS.p", "wb"))

# ------------- end of csv to data -----------------------
'''
# total population
N = 9000000 #36000000 
# population density in /m^2
pd = 0.000109 # 0.000122
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
bnds = list(tuple((0, np.inf) for i in range(1)) + tuple((1*gamma, 20*gamma) for i in range(1)) + tuple((NR[i]/gamma, 1) for i in range(len(mu))) + tuple((0, 1 - NR[i]/gamma) for i in range(len(R_0))))
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
# exit()

print("------------------------")
result = minimize(Mu_R_opt, x0=params, method='SLSQP', constraints=cons, bounds=bnds, options= {'disp':True})
print("------------------------")

# print(result.success, ":", result.message)
print(result)

pickle.dump(result.x, open("../AUS_Bulk_Mu_R_Opt2.p", "wb"))

result.x = pickle.load(open("../AUS_Bulk_Mu_R_Opt2.p", "rb"))

res_epsilon = result.x[0]
res_beta = result.x[1]
res_mu = result.x[2: len(NI)+2]
res_R0 = result.x[len(NI)+2:]

print(np.mean(res_mu), np.std(res_mu))

res_S0 = [NI[i] / (res_mu[i] * res_beta) for i in range(len(res_mu))]
res_I0 = [NR[i] / gamma for i in range(len(NR))]

x_pts = np.arange(1, len(res_I0)+1, 1)
plt.plot(x_pts, res_S0, label="S")
plt.plot(x_pts, res_I0, label="I")
plt.plot(x_pts, res_R0, label="R")
plt.plot(x_pts, res_mu, label="mu")
plt.legend()
plt.show()

for i in range(len(NI)):
    print("-------bulk--------")
    print("beta:", res_beta)
    print("S0:", res_S0[i])
    print("I0:", res_I0[i])
    print("curr I0:", I_0[i])
    print("R0:", res_R0[i])
    print("NI:", NI[i])
    print("NR:", NR[i])
    print("CP:", res_mu[i])
    # input('')
exit()
# ------------------------ for Italy *** end ------------------------
'''


# ---------------------------------------------
res = []
org_I0 = deepcopy(I_0)
for iter in range(20):
    print("----------- sample -----------------", iter)
    
    N = 9000000 #36000000 
    # population density in /m^2
    pd = 0.000109 # 0.000122

    # sample collect
    sample_inf = []
    # sample_rate = 0.0001
    sample_rate = 0.00025
    confidence = 99
    conf_int_z = {90: 1.645, 95: 1.96, 99: 2.576}
    low = True

    Duration = len(I_0)
    for sample_T in range(Duration):
        sample_N = int(sample_rate * N)
        INF = org_I0[sample_T]
    #     inf_dist = np.zeros(N)
    #     print(N, INF)
    #     inf_index = random.sample(range(0, N), INF)
    #     print(sample_T, INF)
    #     for i in inf_index:
    #         inf_dist[i] = 1

    #     sample_index = random.sample(range(0, N), sample_N)
    #     sample_inf.append(len([i for i in sample_index if inf_dist[i] == 1 and random.uniform(0,1) < 0.95]))

    # pickle.dump(sample_inf, open("../AUS_sample_inf2_"+str(iter)+".p", "wb"))
    sample_inf = pickle.load(open("../AUS_sample_inf2_"+str(iter)+".p", "rb"))

    # New Infected Total
    NI = [v[0] / float(N) for k,v in daily_data.items()]
    # New Recovered Total
    NR = [v[1] / float(N) for k,v in daily_data.items()]

    # New Infected Sample
    NI = [v for v in NI[:Duration]]
    # New Recovered
    NR = [v for v in NR[:Duration]]
    # current infected I_0
    I_0 = [i / float(sample_N) for i in sample_inf]

    print("=================")
    for i in I_0:
        print(i)
    # input('')

    # standard error
    std_err_NI = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in NI]
    std_err_NR = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in NR]
    std_err_I0 = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in I_0]

    low_I_0 = [I_0[i] - conf_int_z[confidence] * std_err_I0[i] for i in range(len(I_0))]
    low_NI = [NI[i] - conf_int_z[confidence] * std_err_NI[i] for i in range(len(NI))]
    low_NR = [NR[i] - conf_int_z[confidence] * std_err_NR[i] for i in range(len(NR))]

    high_I_0 = [I_0[i] + conf_int_z[confidence] * std_err_I0[i] for i in range(len(I_0))]
    high_NI = [NI[i] + conf_int_z[confidence] * std_err_NI[i] for i in range(len(NI))]
    high_NR = [NR[i] + conf_int_z[confidence] * std_err_NR[i] for i in range(len(NR))]

    if low:
        # lower limit of I_t
        I_0, NI, NR = low_I_0, low_NI, low_NR
    else:
        # upper limit of I_t
        I_0, NI, NR = high_I_0, high_NI, high_NR

    print("=================")
    for i in I_0:
        print(i)
    # input('')
    # contact sphere in m
    r = 1.8288
    # trip avg
    trip_avg = 1
    # Contact Rate
    C = math.pi * math.pow(r, 2) * pd * trip_avg
    print("C:", C)
    # gamma
    gamma = 0.04788463

    # param mu
    mu = [0.5 for i in range(len(NI))]
    # param R_0
    R_0 = [0.5 for i in range(len(NI))]
    # epsilon
    epsilon = 0.5

    beta = 0.5

    params = [epsilon] + [beta] + mu + R_0

    # bounds for beta, mu, and R_0
    bnds = list(tuple((0, np.inf) for i in range(1)) + tuple((1*gamma, 20*gamma) for i in range(1)) + tuple((NR[i]/gamma, 1) for i in range(len(mu))) + tuple((0, 1 - NR[i]/gamma) for i in range(len(R_0))))
    print("bounds", bnds)
    # exit()
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

    print(len(mu), len(R_0), len(NI))

    print("------------------------")
    result = minimize(Mu_R_opt, x0=params, method='SLSQP', constraints=cons, bounds=bnds, options= {'disp':True})
    print("------------------------")

    # print(result.success, ":", result.message)
    print(result)

    if low:
        pickle.dump(result.x, open("../AUS_sample_Mu_R_Opt_low2_"+str(iter)+".p", "wb"))
        result.x = pickle.load(open("../AUS_sample_Mu_R_Opt_low2_"+str(iter)+".p", "rb"))
    else:
        pickle.dump(result.x, open("../AUS_sample_Mu_R_Opt_high2_"+str(iter)+".p", "wb"))
        result.x = pickle.load(open("../AUS_sample_Mu_R_Opt_high2_"+str(iter)+".p", "rb"))


    res_epsilon = result.x[0]
    res_beta = result.x[1]
    res_mu = result.x[2: len(NI)+2]
    res_R0 = result.x[len(NI)+2:]

    print(np.mean(res_mu), np.std(res_mu))

    res_S0 = [NI[i] / (res_mu[i] * res_beta) for i in range(len(res_mu))]
    res_I0 = [NR[i] / gamma for i in range(len(NR))]

    for i in range(Duration):
        print(res_S0[i] + res_I0[i] + res_R0[i] + res_epsilon, res_epsilon)

    # continue

    Duration = len(res_mu)
    gamma = 0.04788463
    sample = pickle.load(open("../AUS_sample_Mu_R_Opt_low2_"+str(iter)+".p", "rb"))
    sample_mu_low = sample[2: Duration+2]
    sample_beta_low = sample[1]
    sample_R0_low = sample[Duration+2:]
    sample_S0_low = [low_NI[i] / (sample_mu_low[i] * sample_beta_low) for i in range(len(sample_mu_low))]
    sample_I0_low = [low_NR[i] / gamma for i in range(len(low_NR))]

    sample = pickle.load(open("../AUS_sample_Mu_R_Opt_high2_"+str(iter)+".p", "rb"))
    sample_mu_high = sample[2: Duration+2]
    sample_beta_high = sample[1]
    sample_R0_high = sample[Duration+2:]
    sample_S0_high = [high_NI[i] / (sample_mu_high[i] * sample_beta_high) for i in range(len(sample_mu_high))]
    sample_I0_high = [high_NR[i] / gamma for i in range(len(high_NR))]

    Bulk = pickle.load(open("../AUS_Bulk_Mu_R_Opt2.p", "rb"))
    Bulk_mu = Bulk[2: Duration+2]

    print("--------------------------")
    T = 0
    Flipped = 0
    for i in range(Duration):
        # print("\tSample\t\tBulk\tT:", i+1)
        print(sample_mu_low[i], Bulk_mu[i], sample_mu_high[i])
        # T += (1 if sample_mu_low[i] < Bulk_mu[i] < sample_mu_high[i] else 0)
        minimum = min(sample_mu_low[i], sample_mu_high[i])
        maximum = max(sample_mu_low[i], sample_mu_high[i])
        T += (1 if minimum < Bulk_mu[i] < maximum else 0)
        Flipped += (1 if sample_mu_low[i] > sample_mu_high[i] else 0)
        print(i+1, T, Flipped)
        print()
        # print(i+1, T)
        # if (sample_mu_low[i] > sample_mu_high[i]):
        #     print("T:", i+1)
        #     print("-------low--------")
        #     print("beta:", sample_beta_low)
        #     print("S0:", sample_S0_low[i])
        #     print("I0:", sample_I0_low[i])
        #     print("curr I0:", low_I_0[i])
        #     print("R0:", sample_R0_low[i])
        #     print("NI:", low_NI[i])
        #     print("NR:", low_NR[i])
        #     print("CP:", sample_mu_low[i])
        #     print()
        #     print("-------high--------")
        #     print("beta:", sample_beta_high)
        #     print("S0:", sample_S0_high[i])
        #     print("I0:", sample_I0_high[i])
        #     print("curr I0:", high_I_0[i])
        #     print("R0:", sample_R0_high[i])
        #     print("NI:", high_NI[i])
        #     print("NR:", high_NR[i])
        #     print("CP:", sample_mu_high[i])
        #     print()
        #     Flipped += 1
            # input('')
    print(T)
    print(Flipped)
    res.append(T / Duration)
print(res)
print(np.mean(res), np.std(res))

# print("--------low----------")
# for i in range(Duration):
#     print(sample_mu_low[i])
# print("--------high----------")
# for i in range(Duration):
#     print(sample_mu_high[i])
# print("---------nulk---------")
# for i in range(Duration):
#     print(Bulk_mu[i])

'''
print(sample_beta_low, sample_beta_high, Bulk[1])
# exit()
x_pts = np.arange(1, len(sample_mu_low)+1, 1)
# plt.plot(x_pts, sample_S0_low, label="S")
# plt.plot(x_pts, sample_I0_low, label="I")
# plt.plot(x_pts, sample_R0_low, label="R")
# plt.plot(x_pts, sample_mu_low, label="mu")
# plt.legend()
# plt.savefig("Sample_low.png")
# plt.clf()

# x_pts = np.arange(1, len(sample_I0_high)+1, 1)
# plt.plot(x_pts, sample_S0_high, label="S")
# plt.plot(x_pts, sample_I0_high, label="I")
# plt.plot(x_pts, sample_R0_high, label="R")
# plt.plot(x_pts, sample_mu_high, label="mu")
# plt.legend()
# plt.savefig("Sample_high.png")
# plt.clf()

plt.plot(x_pts, sample_mu_high, label="high_mu")
plt.plot(x_pts, Bulk_mu, label="bulk_mu")
plt.plot(x_pts, sample_mu_low, label="low_mu")
plt.legend()
plt.savefig("Bulk_mu_comp"+str(Duration)+".png")
plt.clf()
'''