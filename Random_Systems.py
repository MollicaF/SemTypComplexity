import pandas as pd
from IB.Model import *
np.random.seed(2024)

# Universe (pqrs)

universe = []
for p in [0, 1]:
    for q in [0, 1]:
        for r in [0, 1]:
            for s in [0, 1]:
                universe.append((p, q, r, s))


# Meanings that obey the semantics

p_U_M = np.zeros((len(universe), len(universe)))
for i, ustar in enumerate(universe):
    for j, u in enumerate(universe):
        c = np.sum([x == y for x, y in zip(ustar, u)])
        p_U_M[j, i] = 0.75**c * 0.25**(4-c)

p_U_M = p_U_M / (p_U_M).sum(0)


# Meanings that violate semantics

permutation = np.arange(len(universe))
np.random.shuffle(permutation)
idx = np.empty_like(permutation)
idx[permutation] = np.arange(len(permutation))
z_U_M = p_U_M[:, idx]


# Biased Need Probability 1

p_M = np.array([0.0623, 0.0623, 0.0466, 0.04198, 0.04198, 0.04508, 0.04508, 0.05009,
                0.09046, 0.09046, 0.0786, 0.0786, 0.07158, 0.07158, 0.08209, 0.08209])**3
p_M = p_M / p_M.sum()

# Biased Need Probability 2

z_M = np.arange(1, len(universe) + 1)
np.random.shuffle(z_M)
z_M = (1./z_M)**1.5 / np.sum((1./z_M)**1.5)


source = p_M
meanings = p_U_M
param_string = 'Source_P_Meaning_P'

################################################################
from numpy.random import choice, shuffle


def make_random_q(uni, nterms=None):
    if nterms is None:
        nterms = choice([i for i in range(2, len(uni)-1)], 1)[0]

    extension = [i for i in range(nterms)]
    if len(extension) < len(uni)+1:
        extra = choice(extension, len(uni)-len(extension))
    else:
        extra = []

    enc = extension + list(extra)
    shuffle(enc)

    q = np.zeros((len(uni), nterms))
    q_long = np.zeros((len(uni), len(uni)))
    for u, w in enumerate(enc):
        q[u, w] = 1.
        q_long[u, w] = 1.

    if len(extension) < len(uni) + 1:
        rep = q[:, nterms-1] / (len(extra)+1)
        for i in range(nterms-1, len(uni)):
            q_long[:, i] = rep

    return q, q_long


rsys = []
for _ in range(1000):

    q, q_long = make_random_q(universe)

    r_pp, d_pp = score_q(p_M, p_U_M, q)
    r_pz, d_pz = score_q(p_M, z_U_M, q)
    r_zp, d_zp = score_q(z_M, p_U_M, q)
    r_zz, d_zz = score_q(z_M, z_U_M, q)

    rsys.append([q.T, r_pp, d_pp, r_pz, d_pz, r_zp, d_zp, r_zz, d_zz])

rsys_df = pd.DataFrame(rsys, columns=['Q', 'pp_Rate', 'pp_Distortion',
                                           'pz_Rate', 'pz_Distortion',
                                           'zp_Rate', 'zp_Distortion',
                                           'zz_Rate', 'zz_Distortion'])

rsys_df.to_csv('Random_Systems.csv')
