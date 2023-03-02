import random
from termios import VMIN
from tkinter import Canvas
import numpy as np
from scipy.spatial.distance import *
from copy import deepcopy
import pickle as pickle
import math
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# updates the mean CP of each individual zone due to intrazonal interactions
def update(Mean_CPs, N_zonal):
    Mean_CPs1 = deepcopy(Mean_CPs)
    for i in range(nZ):
        contact = math.pi * math.pow(r, 2) * (float(N_zonal[i]) / area[i])
        # print(alpha_decay, prob[i], contact)
        Mean_CPs1[i] = alpha_decay * Mean_CPs[i] + prob[i] * sum([contact * Mean_CPs[j] for j in range(nZ) if j != i]) #contact * Mean_CPs[i]#

        # print(i, "->", Mean_CPs[i], Mean_CPs1[i])
    return Mean_CPs1

# updates the mean CP of each zone due to interzonal interactions
def merge(Mean_CPs, N_zonal):
    Mean_CPs1 = deepcopy(Mean_CPs)
    N_zonal1 = deepcopy(N_zonal)
    for i in range(nZ):
        Mean_CPs1[i] = ((A[i,i] * N_zonal[i] * Mean_CPs[i] + sum([A[j,i]*N[j]*Mean_CPs[j] for j in range(nZ) if i != j])) 
                                    / (A[i,i] * N_zonal[i] + sum([A[j,i]*N[j] for j in range(nZ) if i != j])))
        
        N_zonal1[i] = ((A[i,i] * N_zonal[i] + sum([A[j,i]*N[j] for j in range(nZ) if i != j])) 
                                    / (A[i,i] + sum([A[j,i] for j in range(nZ) if i != j])))
        
    return Mean_CPs1, N_zonal1

# ---------------------- NYC borough 5 -----------------------------
# no of zones
nZ = 5
# initial population of each zone
N = [1472654, 2736074, 1694263, 2405464, 495747]
# area - sq mi
area = [42.2, 69.4, 22.7, 108.7, 57.5]
# area of each zone in m^2
area = [a * 2600000 for a in area]
# data timepoints
timepoints = 100

r = 1.8288

# Data from optimization
beta, mu, R0 = [], [], []
for r in range(nZ):
    res = pickle.load(open("NYC_mu_opt_z"+str(r)+"1.p", "rb"))

    beta.append(res[1])
    mu.append(res[2: timepoints+2])
    R0.append(res[timepoints+2:])

CPs_zonal = {i : mu[i] for i in range(nZ)}
Mean_CPs = {i: np.mean(CPs_zonal[i]) for i in range(nZ)}
N_zonal = {i: N[i] for i in range(nZ)}

r = 1.8288
sigma, alpha, delta = 0.25, 0.75, 0.025
alpha_decay = 0.815#1 / float(np.sqrt(10))
gamma = [b / 4.5 for b in beta]
C = [math.pi * math.pow(r, 2) * (N[i] / area[i]) for i in range(nZ)]
prob = [beta[i] / C[i] for i in range(nZ)]

# Transition Matrix
A = pickle.load(open("mob_mat_A.p", "rb"))
print("Mobility:", A)

N_zonal_T = []
Mean_CPs_T = []

duration = 30
T = 1
print("T = 0", Mean_CPs, "\n")
while (T <= duration):
    inner_Mean_CPs = update(Mean_CPs, N_zonal)
    Mean_CPs, N_zonal = merge(inner_Mean_CPs, N_zonal)

    Mean_CPs_T.append(Mean_CPs)
    N_zonal_T.append(N_zonal)

    print("T:",T, "->", Mean_CPs,"\n", N_zonal,"\n")
    T += 1

# ------------------ plotting -----------------------
Tested_inf_T = pickle.load(open("nyc_NI1.p", "rb"))

region = {0: 'Bronx', 1: 'Brooklyn', 2: 'Manhattan', 3: 'Queens', 4: 'Staten Island'}
color = ["tomato", "orange", "yellow", "royalblue", "green"]

x_pts = np.arange(len(Mean_CPs_T))
fig, ax1 = plt.subplots()

ax1.set_xlabel("Time in Days", size=13)

for i in range(nZ):
    ax1.plot(x_pts, [(Mean_CPs_T[t][i] * N_zonal_T[t][i] - Tested_inf_T[i][t]) / N_zonal_T[t][i] for t in range(len(x_pts))], label=region[i], color=color[i])

ax1.set_ylabel('Normalized (uN - I)', size=13, color="black")
ax1.legend(loc="upper right")

fig.tight_layout()
plt.savefig("nyc_uN.png", dpi=300)
plt.clf()

x_pts = np.arange(len(Mean_CPs_T))
fig, ax1 = plt.subplots()
for i in range(nZ):
    ax1.plot(x_pts, [Tested_inf_T[i][t] / N[i] for t in range(len(x_pts))], label=region[i], color=color[i])
print(Tested_inf_T)
ax1.set_xlabel('Time in Days', size=13, color="black")
ax1.set_ylabel('Normalized Tested Infected', size=13, color="black")
ax1.legend(loc="upper right")
fig.tight_layout()
plt.savefig("nyc_multi_plt_inf.png", dpi=300)
plt.clf()

x_pts = np.arange(len(Mean_CPs_T))
fig, ax1 = plt.subplots()
for i in range(nZ):
    St = [Mean_CPs_T[t][i] * N_zonal_T[t][i] - Tested_inf_T[i][t] for t in range(len(x_pts))]
    slope = np.diff(St[1:])/np.diff(x_pts[1:])
    ax1.plot(x_pts[2:], slope, label=region[i], color=color[i])

ax1.set_xlabel('Time in Days', size=13, color="black")
ax1.set_ylabel('slope', size=13, color="black")
ax1.legend(loc="upper left")
fig.tight_layout()
plt.savefig("nyc_multi_plt_slope.png", dpi=300)
plt.clf()

# --------------- for diff test rates ------------------
x_pts = np.arange(len(Mean_CPs_T))
data = []
data_slope = []
for i in range(nZ):
    St = [Mean_CPs_T[t][i] * N_zonal_T[t][i] - Tested_inf_T[i][t] for t in range(len(x_pts))]
    slope = np.diff(St[1:])/np.diff(x_pts[1:])
    print(slope)
    data.append(St)
    data_slope.append(slope)
print(np.array(data_slope).shape)

cor = np.corrcoef(np.array(data))
cor_slope = np.corrcoef(np.array(data_slope))
print(cor_slope)

for i in range(nZ):
    for j in range(nZ):
        # if cor_slope[i,j] < 0:
        #     cor_slope[i,j] = -cor_slope[i,j]
        if i == j:
            cor_slope[i,j] = 0

fig, ax = plt.subplots()

cax = ax.matshow(cor_slope, cmap=plt.cm.Blues)
fig.colorbar(cax)

for i in range(nZ):
    for j in range(nZ):
        c = cor_slope[j,i]
        # if c < 0:
        #     c = -c
        ax.text(i, j, "{:.2f}".format(c), va='center', ha='center')

ax.set_xticks([(i) for i in range(nZ)])
ax.set_yticks([(i) for i in range(nZ)])
ax.set_xticklabels([i for i in ['BX', 'BK', 'MN', 'QN', 'SI']])
ax.set_yticklabels([i for i in ['BX', 'BK', 'MN', 'QN', 'SI']])

ax.set_xlabel('Correlation of \u0394(uN-I)', size=13, color="black")
ax.set_ylabel('Correlation of \u0394(uN-I)', size=13, color="black")
ax.legend(loc="upper left")
fig.tight_layout()
plt.savefig("nyc_multi_plt_cor_slope.png", dpi=300)
plt.clf()

# ------------- testrate plot --------------------------------
weekly_tests_bor = [[8711.77,	9815.88,	9300.45,	9232.84,	10557.54],
                    [10685.89,	10988.7,   11607.62,	9965.93,    11049.41],
                    [7646.49,	8496.69,	8057.69,	7450.38,	8087.28],
                    [5954.77,	7398.84,	7144.08,	5634.78,	6070.03]]

x_pts = np.arange(duration)
daily_test_bor = [[0, 0, 0, 0, 0] for i in x_pts]
for week in range(len(weekly_tests_bor)):
    for day in range(7):
        for z in range(nZ):
            daily_test_bor[week*7+day][z] = weekly_tests_bor[week][z] + random.uniform(0,100)
daily_test_bor[-2] = [5954.77,	7398.84,	7144.08,	5634.78,	6070.03]
daily_test_bor[-1] = [5954.77,	7398.84,	7144.08,	5634.78,	6070.03]

print(daily_test_bor)
fig, ax1 = plt.subplots()
for i in range(nZ):
    ax1.plot(x_pts, gaussian_filter1d([daily_test_bor[t][i] / (100000.0) for t in range(len(x_pts))], sigma=2), label=region[i], color=color[i])
print(daily_test_bor)
ax1.set_xlabel('Time in Days', size=13, color="black")
ax1.set_ylabel('Normalized Test Rate', size=13, color="black")
ax1.legend(loc="upper right")
fig.tight_layout()
plt.savefig("nyc_multi_plt_testrate1.png", dpi=300)
plt.clf()

# -------------- positivity rate --------------------

weekly_pos_bor = [[34.71,	41.22,	32.06,	28.88,	37.44,	38.04],
                    [31.68,	35.57,	29.88,	25.25,	35.9, 35.13],
                    [21.23,	21.64,	20.17,	16.83,	25.18,	24.79],
                    [12.71,	11.47,	12.43,	10.3,   15.8	,16.33]]


x_pts = np.arange(duration)
daily_pos_bor = [[0, 0, 0, 0, 0] for i in x_pts]
for week in range(len(weekly_pos_bor)):
    for day in range(7):
        for z in range(nZ):
            daily_pos_bor[week*7+day][z] = weekly_pos_bor[week][z] + random.uniform(0,5)
daily_pos_bor[-2] = [7.4,	6.34,	7.16,	6.27,	9.33,	10.23]
daily_pos_bor[-1] = [7.4,	6.34,	7.16,	6.27,	9.33,	10.23]

print(daily_pos_bor)
fig, ax1 = plt.subplots()
for i in range(nZ):
    ax1.plot(x_pts, gaussian_filter1d([daily_pos_bor[t][i] / N[i] for t in range(len(x_pts))], sigma=2), label=region[i], color=color[i])
print(daily_pos_bor)
ax1.set_xlabel('Time in Days', size=13, color="black")
ax1.set_ylabel('Positivity Rate', size=13, color="black")
ax1.legend(loc="upper right")
fig.tight_layout()
plt.savefig("nyc_multi_plt_posrate.png", dpi=300)
plt.clf()

# -------------- vaccination rate --------------------
daily_vax = [[42, 732, 2867, 2652, 2370, 2304, 2053, 1419, 442, 2023, 1923, 1976, 1875, 2091, 1107, 357, 1031, 1790, 1630, 1242, 1646, 904, 319, 1358, 1323, 1334, 1172, 1374, 103, 270],
            [75, 2078, 4425, 3737, 3269, 3361, 2811, 2353, 1324, 3071, 2445, 2569, 2515, 2882, 1886, 893, 1848, 2264, 2115, 1718, 2088, 1540, 814, 1759, 1725, 1623, 1530, 1618, 205, 634],
            [49, 927, 2016, 2103, 1850, 1827, 1758, 980, 555, 1666, 1576, 1569, 1645, 1736, 853, 532, 1180, 1635, 1425, 1236, 1453, 837, 607, 1278, 1185, 1051, 1030, 1278, 271, 470],
            [100, 2241, 3942, 3375, 2887, 2946, 2817, 2458, 1431, 2739, 2327, 2366, 2337, 2763, 1939, 886, 1664, 1982, 1742, 1542, 1837, 1599, 850, 1629, 1548, 1445, 1265, 1538, 183, 572],
            [19, 303, 757, 685, 609, 568, 479, 356, 262, 550, 603, 420, 571, 512, 330, 137, 363, 364, 351, 311, 314, 239, 144, 257, 267, 231, 224, 257, 24, 96]]

fig, ax1 = plt.subplots()
for i in range(nZ):
    ax1.plot(x_pts, gaussian_filter1d([daily_vax[i][t] / N[i] for t in range(len(x_pts))], sigma=2), label=region[i], color=color[i])
print(daily_vax)
ax1.set_xlabel('Time in Days', size=13, color="black")
ax1.set_ylabel('Normalized Vaccination Rate', size=13, color="black")
ax1.legend(loc="upper right")
fig.tight_layout()
plt.savefig("nyc_multi_plt_vaxrate.png", dpi=300)
plt.clf()

