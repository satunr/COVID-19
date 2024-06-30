import numpy as np
from scipy.stats import entropy
import pickle

def row_entropy(matrix, base=2):
    entropies = []

    for row in matrix:
        # Calculate entropy for each row
        row_entropy = entropy(row, base=base)
        entropies.append(row_entropy)

    # Calculate the mean of all entropies
    mean_entropy = np.mean(entropies)

    return mean_entropy


[_, _, Mixing_Mat_loc] = pickle.load(open("daily_CP_for_dist_grid5000_adj.p", "rb"))
[_, _, Mixing_Mat_snt] = pickle.load(open("daily_CP_for_dist_5000_no_adj2.p", "rb"))
snt_ent = row_entropy(Mixing_Mat_snt)
loc_ent = row_entropy(Mixing_Mat_loc)

print("loc =>",loc_ent)
print("snt =>", snt_ent)
