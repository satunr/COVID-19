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
from scipy import linalg, interpolate

def SIRS(Z):
    global T, R0, gamma, delta, beta, N, prob, contact_rate
    
    print("Day: ", T)

    z0 = deepcopy(Z)

    beta = contact_rate * prob
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

E = [0.95, 0, 0.05, 0, 0]

N = 100000
X, Y = 2500, 2500
pd = float(N) / (X * Y)
# for delta 3.2, alpha 2, omic 9.5
R0 = 3.2
status = ['S', 'E', 'I', 'R', 'D']
gamma, delta = 0.05, 0.025
contact_rate = 0.1
prob = 0.7
beta = contact_rate * prob

Duration = 120
T = 1
Z = [int(each * N) for each in E]

status_counts = []
new_inf = []
new_recov = []

while T <= Duration:
    
    # Simulate an outbreak by changing contact_rate for a short period
    if 10 <= T <= 25:
        contact_rate = 0.7
    else:
        contact_rate = 0.2
    Z, NI, NR = SIRS(Z)

    status_counts.append(Z)
    new_inf.append(NI)
    new_recov.append(NR)

    T += 1

# pickle.dump([status_counts, new_inf, new_recov], open("../SIR_delta_outbreak.p", "wb"))

# -------------- Mu R Optimization (True Mean CP)-------------------
def Mu_R_opt(params, *args):
    epsilon = params[0]
    # print("E:", epsilon)
    return epsilon

'''
# New Infected
NI = [v / float(N) for v in new_inf]
# New Recovered
NR = [v / float(N) for v in new_recov]
# current infected I_0
I_0 = [i[2] / float(N) for i in status_counts]
print("------ NI bulk--------")
for i in range(len(NI)):
    print(i, NI[i])
print("------ NR bulk --------")
for i in range(len(NR)):
    print(i, NR[i])
print("------ I_0 bulk--------")
for i in range(len(I_0)):
    print(i, I_0[i])

# contact sphere in m
r = 1.8288
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

pickle.dump(result.x, open("../Bulk_Mu_R_Opt_outbreak.p", "wb"))
result.x = pickle.load(open("../Bulk_Mu_R_Opt_outbreak.p", "rb"))

res_epsilon = result.x[0]
res_beta = result.x[1]
res_mu = result.x[2: len(NI)+2]
res_R0 = result.x[len(NI)+2:]

print(np.mean(res_mu), np.std(res_mu))

res_S0 = [NI[i] / (res_mu[i] * res_beta) for i in range(len(res_mu))]
res_I0 = [NR[i] / gamma for i in range(len(NR))]

for i in res_mu:
    print(i)
exit()
'''
# ---------------------------------------------
[status_counts, new_inf, new_recov] = pickle.load(open("../SIR_delta_outbreak.p", "rb"))

# confidence interval calculation
value = []
for iter in range(20):
    # sample collect
    rate = 20
    sample_inf = []
    sample_rate = rate / 100.0
    Duration = 120
    interval = 16
    sample_pts = [i*interval for i in range(int(np.ceil(Duration / interval)))]

    Test_Eff = 0.95
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
        I_0 = [low_I_0[t] for t in sample_pts]
        NI = [low_NI[t] for t in sample_pts]
        NR = [low_NR[t] for t in sample_pts]
    else:
        # upper limit of I_t
        I_0 = [high_I_0[t] for t in sample_pts]
        NI = [high_NI[t] for t in sample_pts]
        NR = [high_NR[t] for t in sample_pts]

    
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


# plotting and mse calculation
Duration = 120
result = pickle.load(open("../Bulk_Mu_R_Opt_outbreak.p", "rb"))
[status_counts, new_inf, new_recov] = pickle.load(open("../SIR_delta_outbreak.p", "rb"))

True_Mean_CPs = result[2: Duration+2]

for iter in range(1):
    low_res = pickle.load(open("../sample_Mu_R_Opt_low"+str(iter)+".p", "rb"))
    high_res = pickle.load(open("../sample_Mu_R_Opt_high"+str(iter)+".p", "rb"))


    conf_int1 = []
    conf_int2 = []
    conf_int8 = []
    conf_int16 = []

    int1_x = np.arange(1, len(True_Mean_CPs)+1, 1)
    int2_x = np.arange(1, len(True_Mean_CPs)+1, 2)
    int8_x = np.arange(1, len(True_Mean_CPs)+1, 8)
    int16_x = np.arange(1, len(True_Mean_CPs)+1, 16)
    res_mu_low = low_res[2: len(int16_x)+2]
    res_mu_high = high_res[2: len(int16_x)+2]

    count = 0
    for i in range(len(res_mu_low)):
        if res_mu_low[i] > res_mu_high[i]:
            conf_int16.append((res_mu_high[i], res_mu_low[i]))
        else:
            conf_int16.append((res_mu_low[i], res_mu_high[i]))

    count = 0
    for i in int16_x:
        # print(conf_int1[i-1][0], True_Mean_CPs[i-1], conf_int1[i-1][1])
        if conf_int16[(i-1)//16][0] <= True_Mean_CPs[i-1] <= conf_int16[(i-1)//16][1]:
            count += 1
    print(count)
    print(conf_int16)
# exit()

conf_int2 = [(0.47423350232779565, 0.48046454315042314), (0.47005413283945846, 0.4761941376762778), (0.46616283236176786, 0.4719365207776606), (0.46337303721905504, 0.4685488263496807), (0.4629072911457786, 0.4673915910938311), (0.43938489507061274, 0.4618843458535066), (0.5506494939883374, 0.5937910340149227), (0.6891429576972654, 0.7453137641911379), (0.750460109563383, 0.7795606550434264), (0.6885323629285629, 0.8391065138199646), (0.6571598451109091, 0.754404792586339), (0.6271116820157129, 0.7259946379071622), (0.5869734402785917, 0.6837406671796604), (0.538105725674467, 0.6314750418983606), (0.5002503999421994, 0.5789856096367118), (0.4993081565640788, 0.532980066561459), (0.4979004343707442, 0.5012429531217352), (0.4962954796364855, 0.4995489781699169), (0.4946266709445343, 0.49761972611857186), (0.49445123411758934, 0.49611258425621685), (0.4923323569789848, 0.49409717801878783), (0.49174865156846137, 0.492846629825356), (0.48986371432665393, 0.4909115503450554), (0.49038027129532896, 0.49060076056164525), (0.4897850658333028, 0.4899628892777295), (0.48734497101098634, 0.48734880110500706), (0.4874615649100184, 0.4880343527683134), (0.48596644313199344, 0.4866038844893263), (0.48523011201291616, 0.48608624434029873), (0.4837917134582955, 0.4846720196271289), (0.4829049760669061, 0.4839019588397538), (0.48179900164241, 0.4828359810426833), (0.4793477390888982, 0.4800769890436825), (0.482610361001467, 0.4842204931199158), (0.48143411006143727, 0.4830072696827829), (0.4787880384844589, 0.4800097200565675), (0.48260782410288866, 0.4845831060396009), (0.4812069167968054, 0.4830698963670113), (0.47955494663456416, 0.48122927192724435), (0.4785369716354694, 0.48010196595373883), (0.4773965123389392, 0.4787869996517398), (0.4772394276581783, 0.4786320256362913), (0.4776023550447855, 0.4790769635035501), (0.47618548533274446, 0.4773696303652224), (0.47762259014968483, 0.47909025042504866), (0.476621873476794, 0.47786428923243934), (0.4762250559093404, 0.47734494579175885), (0.47677193963428405, 0.47796389621160784), (0.47678735036860853, 0.47792312134714326), (0.47689506143652877, 0.47799473496892336), (0.4772056756788404, 0.4783028592676575), (0.4775146842686096, 0.47860957026749296), (0.47739338490253813, 0.47837692801163356), (0.47736686797244754, 0.47825827708325014), (0.47717161848197676, 0.4779266910438786), (0.4746824553816808, 0.4747282889129978), (0.4765889830963162, 0.476986107960946), (0.47683286671495195, 0.4772005193632334), (0.477586226025921, 0.4780552476895264), (0.477021810113052, 0.47722967907244007)]
conf_int8 = [(0.473200582774548, 0.47933683683352624), (0.4669005540975192, 0.4713430122056927), (0.7430930155644253, 0.7691292028664681), (0.5874993022896178, 0.684504472617453), (0.498050686455291, 0.501062066978467), (0.49332707130013737, 0.4948011736464025), (0.48690172025862866, 0.487892013572932), (0.48414659121489934, 0.4843392288204086), (0.48416388145358374, 0.4855408104906067), (0.4797439057927913, 0.4808897597004056), (0.4791993088822179, 0.480540794444234), (0.48133725935517274, 0.48308315034164434), (0.47782120180748816, 0.4787432913202832), (0.47577639786134374, 0.4758206885607238), (0.4779063636253481, 0.4781694312826599)]
conf_int16 = [(0.4719285469772196, 0.4782065814903202), (0.7320091534870838, 0.7639218629427892), (0.49808680511991765, 0.5013833474582625), (0.4890694626038726, 0.48926642158385947), (0.48263255107943, 0.48388143519170335), (0.47853888922887006, 0.4799311388017054), (0.4788460025903512, 0.4801820219597834), (0.47878208966726005, 0.47949535743130967)]
print(len(conf_int2))

# interpolate and get approzimate daily intervals
int_x, conf_int = int16_x, conf_int16
low_f = interpolate.interp1d(int_x, [v[0] for v in conf_int])
high_f = interpolate.interp1d(int_x, [v[1] for v in conf_int])

mse_values = []
for i in int1_x:
    try:
        l, h = low_f(i), high_f(i)
        mse_values.append((((l+h)/2) - True_Mean_CPs[i-1])**2)
    except:
        continue
mse = np.mean(mse_values)


fig, ax = plt.subplots()

mean_x = np.arange(1, len(True_Mean_CPs)+1, 1)
ax.plot(mean_x, True_Mean_CPs, label="True Mean CP", color="cornflowerblue")
# ax2.scatter(int2_x, [status_counts[t-1][2] for t in int2_x], label="Daily Infected", color="Tomato", s=7)

int2_x = np.arange(1, len(True_Mean_CPs)+1, 2)
int8_x = np.arange(1, len(True_Mean_CPs)+1, 8)
int16_x = np.arange(1, len(True_Mean_CPs)+1, 16)

# ax.fill_between(int1_x, [f[0] for f in conf_int1], [f[1] for f in conf_int1], color='green', alpha=0.15, label='Sample CP Interval (1 Days)')
# for line_x in int1_x:
#     plt.axvline(x=line_x, color='green', linestyle='--', alpha=0.15)

# ax.fill_between(int2_x, [f[0] for f in conf_int2], [f[1] for f in conf_int2], color='green', alpha=0.15, label='Sample CP Interval (2 Days)')
# for line_x in int2_x:
#     plt.axvline(x=line_x, color='green', linestyle='--', alpha=0.15)

# ax.fill_between(int8_x, [f[0] for f in conf_int8], [f[1] for f in conf_int8], color='purple', alpha=0.15, label='Sample CP Interval (8 Days)')
# for line_x in int8_x:
#     plt.axvline(x=line_x, color='purple', linestyle='--', alpha=0.15)

ax.fill_between(int16_x, [f[0] for f in conf_int16], [f[1] for f in conf_int16], color='tomato', alpha=0.25, label='Sample CP Interval (16 Days)')
for line_x in int16_x:
    plt.axvline(x=line_x, color='tomato', linestyle='--', alpha=0.25)

ax.set_xlabel("Days", size=14)
ax.set_ylabel("Mean CP", color="blue", size=14)
# ax2.set_ylabel("Infected", color="red", size=14)


ax.legend(loc='upper right')
# ax2.legend(loc='upper right', bbox_to_anchor=(1, .85))
plt.title("Mean Squared Error: {val:.6f}".format(val=mse), size=16)
plt.tight_layout()
plt.savefig("../Figs_jpg/SIM_PLOT_BULK16.jpg", dpi=300)
plt.show()
plt.clf()
