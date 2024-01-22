"""
The goal here is to take an encoder a beta schedule and output the rate, distortion, and q on the optimal frontier
"""
import numpy as np


def log2_single(x):
    if x == 0: return 0.0
    else: return np.log2(x)


log2 = np.vectorize(log2_single)


def mask(x):
    mask = np.zeros((x.shape[0], x.shape[0]))
    for i, j in enumerate(x):
        mask[i, j] = 1.
    return mask[:, ~np.all(mask == 0, axis=0)]


def mergecols(q, tol=0.01):
    """
    merge cols that represent near-identical words
    """
    zerotol = 1e-20
    # convert from p(x,w) to p(x|w)
    colsums = np.sum(q, axis=0, keepdims=True)
    q = q[:, np.where(colsums > zerotol)[1]]
    q = q / np.sum(q)
    qn = q / np.sum(q, axis=0, keepdims=True)
    colorder = qn[0, :].argsort()
    # reorder so columns to be merged are adjacent
    q = q[:, colorder]
    qn = qn[:, colorder]
    qn_diff = np.diff(qn, n=1, axis=1)
    qn_diffs = np.max(abs(qn_diff), axis=0)

    qm = q[:, 0:1]
    for i in np.arange(len(qn_diffs)):
        if qn_diffs[i] < tol:
            qm[:, -1:] = qm[:, -1:] + q[:, i + 1:i + 2]
        else:
            qm = np.c_[qm, q[:, i + 1:i + 2]]

    return qm


def decode(q_t_x, p_x, p_y_x):
    p_xy = np.divide(p_y_x * p_x[:, np.newaxis], (p_y_x * p_x[:, np.newaxis]).sum(),
                     out=np.zeros((p_y_x * p_x[:, np.newaxis]).shape),
                     where=(p_y_x * p_x[:, np.newaxis]).sum() != 0)
    q_t = np.matmul(p_x, q_t_x)
    q_y_t = np.divide(np.matmul(q_t_x.T, p_xy), q_t[:, np.newaxis],
                      out=np.zeros(np.matmul(q_t_x.T, p_xy).shape),
                      where=q_t[:, np.newaxis] != 0)
    return q_y_t


def stochastic_bottleneck(p_x, p_y_x, q_t_x, beta, maxiters=10000, verbose=False):
    """
    The stochastic information bottleneck.
        The distortion metric is the cross-entropy
    INPUT:
    OUTPUT:
    """
    iters = 0
    d0 = 0

    p_xy = np.divide(p_y_x * p_x[:, np.newaxis], (p_y_x * p_x[:, np.newaxis]).sum(),
                     out=np.zeros((p_y_x * p_x[:, np.newaxis]).shape),
                     where=(p_y_x * p_x[:, np.newaxis]).sum() != 0)
    q_t = np.matmul(p_x, q_t_x)
    q_y_t = np.divide(np.matmul(q_t_x.T, p_xy), q_t[:, np.newaxis],
                      out=np.zeros(np.matmul(q_t_x.T, p_xy).shape),
                      where=q_t[:, np.newaxis] != 0)
    d = -1*np.matmul(p_y_x, log2(q_y_t.T))

    while not np.isclose(np.matmul(p_x, q_t_x * d).sum(), d0) and iters < maxiters:
        iters += 1
        d0 = np.matmul(p_x, q_t_x * d).sum()
        q_xt = q_t * np.exp2(-1*beta * d)
        q_xt = np.divide(q_xt, q_xt.sum(),
                         out=np.zeros(q_xt.shape),
                         where=q_xt.sum() != 0)
        q_t_x = np.divide(q_xt, q_xt.sum(axis=1, keepdims=True),
                          out=np.zeros(q_xt.shape),
                          where=q_xt.sum(axis=1, keepdims=True) != 0)
        q_t = np.matmul(p_x, q_t_x)
        q_y_t = np.divide(np.matmul(q_t_x.T, p_xy), q_t[:, np.newaxis],
                          out=np.zeros(np.matmul(q_t_x.T, p_xy).shape),
                          where=q_t[:, np.newaxis] != 0)
        d = -1 * np.matmul(p_y_x, log2(q_y_t.T))
        if verbose:
            print(iters, (p_xy * d).sum(), d0)

    q_xt = q_t_x * p_x[:, np.newaxis]
    rate = (q_xt * log2(np.divide(q_xt, np.outer(p_x, q_t),
                                  out=np.zeros(q_xt.shape),
                                  where=np.outer(p_x, q_t) != 0))).sum()
    distortion = np.matmul(p_x, q_t_x * (d + (np.matmul(p_y_x, log2(p_y_x.T)) * np.eye(p_x.shape[0])).sum(axis=1, keepdims=True))).sum()

    return q_t_x, rate, distortion


def score_q(p_x, p_y_x, q_t_x):
    p_xy = (p_y_x * p_x[:, np.newaxis]) / (p_y_x * p_x[:, np.newaxis]).sum()
    q_xt = q_t_x * p_x[:, np.newaxis]
    q_t = np.matmul(p_x, q_t_x)
    q_y_t = np.matmul(q_t_x.T, p_xy) / q_t[:, np.newaxis]
    rate = (q_xt * log2(q_xt / (q_xt.sum(axis=0, keepdims=True) * q_xt.sum(axis=1, keepdims=True)))).sum()
    d = -1 * np.matmul(p_y_x, log2(q_y_t.T))
    distortion = np.matmul(p_x, q_t_x * (
                d + (np.matmul(p_y_x, log2(p_y_x.T)) * np.eye(p_x.shape[0])).sum(axis=1, keepdims=True))).sum()
    return rate, distortion


def deterministic_bottleneck(p_x, p_y_x, f_x, beta, maxiters=10000, verbose=False):
    """
    The stochastic information bottleneck.
        The distortion metric is the cross-entropy
    INPUT:
    OUTPUT:
    """
    iters = 0
    d0 = 0

    p_xy = (p_y_x * p_x[:, np.newaxis]) / (p_y_x * p_x[:, np.newaxis]).sum()
    q_t = np.matmul(p_x, f_x)
    q_y_t = np.matmul(f_x.T, p_xy) / q_t[:, np.newaxis]
    d = -1 * np.matmul(p_y_x, log2(q_y_t.T))
    l = log2(q_t) - beta * d

    while not np.isclose(np.matmul(p_x, f_x * d).sum(), d0) and iters < maxiters:
        iters += 1
        d0 = np.matmul(p_x, f_x * d).sum()
        f_x = mask(np.argmax(l, axis=1))
        q_t = np.matmul(p_x, f_x)
        q_y_t = np.matmul(f_x.T, p_xy) / q_t[:, np.newaxis]

        d = -1 * np.matmul(p_y_x, log2(q_y_t.T))
        l = log2(q_t) - beta * d
        if verbose:
            print(iters, np.matmul(p_x, f_x * d).sum(), d0)

    rate = -1 * (q_t * log2(q_t)).sum()
    distortion = np.matmul(p_x, f_x * (d + (np.matmul(p_y_x, log2(p_y_x.T)) * np.eye(p_x.shape[0])).sum(axis=1, keepdims=True))).sum()

    return f_x, rate, distortion


def rev_deterministic_annealing_IB(p_x, p_y_x, schedule, init, deterministic=False, maxiters=10000, tol=0.05, verbose=False):
    rates = []
    distortions = []
    qs = []

    # First pass
    if deterministic:
        q, r, d = deterministic_bottleneck(p_x, p_y_x, init, schedule[0], maxiters=maxiters)
    else:
        q, r, d = stochastic_bottleneck(p_x, p_y_x, init, schedule[0], maxiters=maxiters)
    rates.append(r)
    distortions.append(d)
    qs.append(q)

    if verbose:
        print(np.round(r, 4), np.round(d, 4), mergecols(q, tol=tol).shape[1])

    for b in schedule[1:]:
        if deterministic:
            q, r, d = deterministic_bottleneck(p_x, p_y_x, q, b, maxiters=maxiters)
        else:
            q, r, d = stochastic_bottleneck(p_x, p_y_x, q, b, maxiters=maxiters)
        rates.append(r)
        distortions.append(d)
        qs.append(q)
        if verbose:
            print(np.round(r, 4), np.round(d, 4), mergecols(q, tol=tol).shape[1])

    return qs, rates, distortions


def achieve_capacity(p_y_x, r_x, maxiters=10000, verbose=False):
    r0 = np.zeros(p_y_x.shape[0])
    iters = 0
    while not np.all(np.isclose(r_x, r0)) and iters < maxiters:
        iters += 1
        r0 = r_x
        q_xy = p_y_x * r_x[:, np.newaxis] / (p_y_x * r_x[:, np.newaxis]).sum()
        q_x_y = q_xy / np.sum(q_xy, axis=0, keepdims=True)
        r_x = np.prod(np.power(q_x_y, p_y_x), axis=1)
        r_x = r_x / r_x.sum()
        if verbose:
            print(iters, np.round(((r_x-r0)**2).sum()**0.5, 10))

    capacity = (q_xy * log2(q_xy / np.outer(r_x, q_xy.sum(axis=0)))).sum()

    return r_x, capacity


def average_capacity(qs, freq, rinit, maxiters=10000, verbose=False):
    p_x = np.zeros(rinit.size)
    for f, q in zip(freq, qs):
        r_x, _ = achieve_capacity(q, rinit, maxiters=maxiters, verbose=verbose)
        p_x += f * r_x
    return p_x / p_x.sum()


#######################################################################################################################
if __name__ == '__main__':

    # Initializations
    p_x = np.array([0.07723991, 0.05938659, 0.1373735, 0.475, 0.12382667, 0.08032, 0.04685333])

    p_y_x = np.array([[1.00e+00, 5.00e-01, 2.50e-01, 2.50e-02, 2.50e-03, 1.25e-03, 6.25e-04],
                      [5.00e-01, 1.00e+00, 5.00e-01, 5.00e-02, 5.00e-03, 2.50e-03, 1.25e-03],
                      [2.50e-01, 5.00e-01, 1.00e+00, 1.00e-01, 1.00e-02, 5.00e-03, 2.50e-03],
                      [2.50e-02, 5.00e-02, 1.00e-01, 1.00e+00, 1.00e-01, 5.00e-02, 2.50e-02],
                      [2.50e-03, 5.00e-03, 1.00e-02, 1.00e-01, 1.00e+00, 5.00e-01, 2.50e-01],
                      [1.25e-03, 2.50e-03, 5.00e-03, 5.00e-02, 5.00e-01, 1.00e+00, 5.00e-01],
                      [6.25e-04, 1.25e-03, 2.50e-03, 2.50e-02, 2.50e-01, 5.00e-01, 1.00e+00]])

    p_y_x = p_y_x / p_y_x.sum(axis=1, keepdims=True)

    p_xy = (p_y_x * p_x[:, np.newaxis]) / (p_y_x * p_x[:, np.newaxis]).sum()

    # Stochastic Bottleneck

    eps = 1e-1
    qinit = (1 - eps) * np.eye(p_x.shape[0]) + eps * np.ones(p_y_x.shape)

    q, r, dist = stochastic_bottleneck(p_x, p_y_x, qinit, beta=2.)

    q = mergecols(q)
    q = q / q.sum(axis=1, keepdims=True)

    print('########', r, dist, '\n', np.round(q, 2))
    score = score_q(p_x, p_y_x, q)
    print('########', score[0], score[1])

    # Deterministic Bottleneck

    finit = np.eye(p_x.shape[0])
    q, r, dist = deterministic_bottleneck(p_x, p_y_x, finit, beta=2.)

    print('########', r, dist, '\n', np.round(q, 2))
    score = score_q(p_x, p_y_x, q)
    print('########', score[0], score[1])

    # Reverse deterministic annealing

    schedule = [2**x for x in np.arange(3, 0, -0.01)]
    qs, rs, ds, = rev_deterministic_annealing_IB(p_x, p_y_x, schedule, finit, deterministic=True, verbose=True)
    print('##############################################################')
    qs, rs, ds, = rev_deterministic_annealing_IB(p_x, p_y_x, schedule, qinit, deterministic=False, verbose=True)

    # Capacity Achieving Prior

    r_x = np.ones(p_y_x.shape[0]) / p_y_x.shape[0]
    _, cap = achieve_capacity(p_y_x, r_x)

    print('Actual Need Rate: ', (p_xy * log2(p_xy / np.outer(p_x, p_xy.sum(axis=0)))).sum())
    print('Channel Capacity: ', cap)

