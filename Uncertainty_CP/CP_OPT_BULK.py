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

def SIRS(Z):
    global T, R0, gamma, delta, beta, N
    
    print("Day: ", T)

    z0 = deepcopy(Z)

    # New Infected
    NI = int((beta * z0[0] * z0[2]) / N)
    NR = int(gamma * z0[2])

    update0 = - int((beta * z0[0] * z0[2]) / N) + int(delta * z0[3])
    update2 = int((beta * z0[0] * z0[2]) / N) - int(gamma * z0[2])
    update3 = int(gamma * z0[2]) - int(delta * z0[3])

    z0[0] = z0[0] + update0
    z0[2] = z0[2] + update2
    z0[3] = z0[3] + update3

    print("Day: ", T, "Pop=", np.sum(z0), z0, '\n')

    return z0, NI, NR 

E = [0.9, 0, 0.1, 0, 0]

N = 1000000
X, Y = 24750, 24750
# N = 5000
# X, Y = 1750, 1750
pd = float(N) / (X * Y)
# for delta 3.2, alpha 2, omic 9.5
R0 = 3.2
status = ['S', 'E', 'I', 'R', 'D']
gamma, delta = 0.05, 0.025
beta = gamma * R0

Duration = 60
T = 1
Z = [int(each * N) for each in E]

status_counts = []
new_inf = []
new_recov = []

while T <= Duration:

    Z, NI, NR = SIRS(Z)

    status_counts.append(Z)
    new_inf.append(NI)
    new_recov.append(NR)

    T += 1

# pickle.dump([status_counts, new_inf, new_recov], open("../SIR_delta.p", "wb"))

# -------------- Mu R Optimization -------------------
def Mu_R_opt(params, *args):
    epsilon = params[0]
    # print("E:", epsilon)
    return epsilon


[status_counts, new_inf, new_recov] = pickle.load(open("../SIR_delta.p", "rb"))

value = []
for iter in range(20):
    # sample collect
    rate = 30
    sample_inf = []
    sample_rate = rate / 100.0
    Duration = 60

    Test_Eff = 1#0.95
    conf_int_z = {90: 1.645, 95: 1.96, 99: 2.576}
    confidence = 95 
    low = True

    for sample_T in range(Duration):
        sample_N = int(sample_rate * N)
        INF = new_inf[sample_T]
    #     inf_dist = np.zeros(N)
    #     inf_index = random.sample(range(0, N), INF)
    #     for i in inf_index:
    #         inf_dist[i] = 1

    #     sample_index = random.sample(range(0, N), sample_N)
    #     # sample_index = [i for i in range(iter*sample_N, (iter+1)*sample_N, 1)]
    #     sample_inf.append(len([i for i in sample_index if inf_dist[i] == 1 and random.uniform(0,1) < Test_Eff]))

    # pickle.dump(sample_inf, open("../sample_inf"+str(rate)+"_"+str(iter)+".p", "wb"))
    sample_inf = pickle.load(open("../sample_inf"+str(rate)+"_"+str(iter)+".p", "rb"))

    # initial infected fraction
    init_I_0 = 0.1
    # New Infected
    NI = [i / (float(sample_N)) for i in sample_inf]
    # New Recovered
    NR = [v / float(N) for v in new_recov[:Duration]]
    # current infected I_0
    I_0 = [sum(NI[:i]) - sum(NR[:i]) + init_I_0 for i in range(Duration)]

    count = 0
    # for i in NI:
    #     print(i)
    #     if (i < 0.00375):
    #         count += 1
    # print(count, "skip")
    # input('')

    # standard error
    std_err_NI = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in NI]
    std_err_NR = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in NR]
    std_err_I0 = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in I_0]

    # print(std_err_I0)
    print("------------------------")
    for i in I_0:
        print(i)
    # input('')

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


    print("------ NI --------")
    for i in range(len(NI)):
        print(i, NI[i])
    print("------ NR --------")
    for i in range(len(NR)):
        print(i, NR[i])
    print("------ I_0 --------")
    for i in range(len(I_0)):
        print(i, I_0[i])
    
    # contact sphere in m
    r = 1.8288
    # trip avg
    trip_avg = 0.7
    # Contact Rate
    C = math.pi * math.pow(r, 2) * pd * trip_avg
    print("C:", C)
    # gamma
    gamma = 0.05

    # param mu
    mu = [0.5 for i in range(len(NI))]
    # param R_0
    R_0 = [0.5 for i in range(len(NI))]
    # epsilon
    epsilon = 0.5

    params = [epsilon] + [beta] + mu + R_0

    # bounds for beta, mu, and R_0
    bnds = list(tuple((0, np.inf) for i in range(1)) + tuple((9*gamma, 10*gamma) for i in range(1)) + 
                tuple((NR[i]/gamma, 1) for i in range(len(mu))) + tuple((0, 1 - NR[i]/gamma) for i in range(len(R_0))))
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
    res_mu = result.x[2: len(NI)+2]
    for i in res_mu:
        print(i)
    
    if low:
        pickle.dump(result.x, open("../sample_Mu_R_Opt_low"+str(iter)+".p", "wb"))
        result.x = pickle.load(open("../sample_Mu_R_Opt_low"+str(iter)+".p", "rb"))
    else:
        pickle.dump(result.x, open("../sample_Mu_R_Opt_high"+str(iter)+".p", "wb"))
        result.x = pickle.load(open("../sample_Mu_R_Opt_high"+str(iter)+".p", "rb"))



    res_epsilon = result.x[0]
    res_beta = result.x[1]
    res_mu = result.x[2: len(NI)+2]
    res_R0 = result.x[len(NI)+2:]


    print(np.mean(res_mu), np.std(res_mu))

    res_S0 = [NI[i] / (res_mu[i] * beta) for i in range(len(res_mu))]
    res_I0 = [NR[i] / gamma for i in range(len(NR))]

    # exit()

    # std_err_mu = [np.abs(np.sqrt(p * (1 - p) / sample_N )) for p in res_mu]

    # sample_mu_low = [res_mu[i] - conf_int_z[confidence] * std_err_mu[i] for i in range(len(res_mu))]

    # sample_mu_high = [res_mu[i] + conf_int_z[confidence] * std_err_mu[i] for i in range(len(res_mu))]

    Duration = 60

    sample = pickle.load(open("../sample_Mu_R_Opt_low"+str(iter)+".p", "rb"))
    sample_mu_low = sample[2: Duration+2]

    sample = pickle.load(open("../sample_Mu_R_Opt_high"+str(iter)+".p", "rb"))
    sample_mu_high = sample[2: Duration+2]

    Bulk = pickle.load(open("../Bulk_Mu_R_Opt.p", "rb"))
    Bulk_mu = Bulk[2: Duration+2]

    print("--------------------------")
    T = 0
    flipped = 0
    for i in range(Duration):
        print("\tSamp Low\tBulk\t\t\tSamp high\tT:", i+1)
        print(sample_mu_low[i], Bulk_mu[i], sample_mu_high[i])
        # print(res_mu[i], Bulk_mu[i])
        minimum = min(sample_mu_low[i], sample_mu_high[i])
        maximum = max(sample_mu_low[i], sample_mu_high[i])
        T += (1 if minimum < Bulk_mu[i] < maximum else 0)
        # T += (1 if sample_mu_low[i] < Bulk_mu[i] < sample_mu_high[i] else 0)
        flipped += (1 if sample_mu_low[i] > sample_mu_high[i] else 0)
        print(i+1, T, flipped)
        print()
    
    print("===>", T, count, (Duration - count), T / (Duration - count))

    if (T / (Duration - count) > 1):
        continue
    # value.append(T / Duration)
    value.append(T / (Duration - count))
    # input('')

print(value)
print(np.mean(value), np.std(value))
# pickle.dump(value, open("bulk_samp_frac_samp20_1.p", "wb"))

