import pickle
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

L_120_7 = pickle.load(open('change4-120-point7.p', 'rb'))
L_135_6 = pickle.load(open('change4-135-point6.p', 'rb'))
L_135_7 = pickle.load(open('change4-135-point7.p', 'rb'))
L_135_8 = pickle.load(open('change4-135-point8.p', 'rb'))
L_150_7 = pickle.load(open('change4-150-point7.p', 'rb'))

# L_120_8 = pickle.load(open('change4-120-point8.p', 'rb'))

L_120_7 = [val[1] for val in L_120_7]
L_135_7 = [val[1] for val in L_135_7]
L_135_6 = [val[1] for val in L_135_6]
L_135_8 = [val[1] for val in L_135_8]
L_150_7 = [val[1] for val in L_150_7]

# L_120_8 = [val[1] for val in L_120_8]

# plt.plot([i for i in range(len(L_120_6))], savgol_filter(L_120_6, 29, 2), label = '120_6')

# plt.plot([i for i in range(len(L_120_7))], savgol_filter(L_120_7, 29, 2), label = '120_7')
plt.plot([i for i in range(len(L_135_7))], savgol_filter(L_135_6, 29, 2), label = '135_6')
plt.plot([i for i in range(len(L_135_7))], savgol_filter(L_135_7, 29, 2), label = '135_7')
plt.plot([i for i in range(len(L_135_8))], savgol_filter(L_135_8, 29, 2), label = '135_8')
# plt.plot([i for i in range(len(L_150_7))], savgol_filter(L_150_7, 29, 2), label = '150_7')

# plt.plot([i for i in range(len(L_120_6))], savgol_filter(L_120_8, 29, 2), label = '120_8')

plt.legend()
plt.show()
