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


params = pd.DataFrame({
    'Element': universe,
    'Source': source,
    'Meaning': ['_'.join([str(x) for x in meanings[:, i]]) for i in range(len(universe))]
})

params.to_csv('Params_'+param_string+'.csv')

eps = 1e-1
qinit = (1 - eps) * np.eye(source.shape[0]) + eps * np.ones(meanings.shape)

schedule = [2 ** x for x in np.arange(4, 0, -0.005)]  # Psource Pmeanings
#schedule = [2 ** x for x in np.arange(8, 0, -0.005)]  # Zsource Pmeanings
#schedule = [2 ** x for x in np.arange(8, 0, -0.005)]  # Zsource Zmeanings
#schedule = [2 ** x for x in np.arange(4, 0, -0.005)]  # Psource Zmeanings

dqs, drs, dds, = rev_deterministic_annealing_IB(source, meanings, schedule, qinit, deterministic=True, verbose=True)
print('##########################################################\n\n')
qs, rs, ds, = rev_deterministic_annealing_IB(source, meanings, schedule, qinit, deterministic=False, verbose=True)

uniq = []
i = 17
for q in dqs:
    if q.shape[1] < i:
        i = q.shape[1]
        uniq.append(q)
        print(q)


ib_front = pd.DataFrame(
    {
        'Beta': schedule,
        'Q': qs,
        'Rate': rs,
        'Distortion': ds
    }
)

dib_front = pd.DataFrame(
    {
        'Beta': schedule,
        'Q': [q.T for q in dqs],
        'Rate': drs,
        'Distortion': dds
    }
)

ib_front.to_csv('Stochastic_' + param_string + '.csv')
dib_front.to_csv('Deterministic_' + param_string + '.csv')
