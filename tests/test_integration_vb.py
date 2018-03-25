import numpy as np
from vbkf.vb_update import sample, update


def get_Q_k(k, T, q, d_t):
    pref = np.array([6.5 + 0.5 * np.cos(np.pi * k / T)])
    mtx = np.array([[(d_t ** 3) / 3, 0., (d_t ** 2) / 2, 0.],
                    [0., (d_t ** 3) / 3., 0., (d_t ** 2) / 2],
                    [(d_t ** 2) / 2, 0., d_t, 0.],
                    [0., (d_t ** 2) / 2, 0., d_t]])
    return (pref * q) @ mtx


def get_R_k(k, T, r):
    pref = np.array([0.1 + 0.5 * np.cos(np.pi * k / T)])
    mtx = np.array([
        [1, 0.5],
        [0.5, 1]
    ])
    return (pref * r) @ mtx


def get_true_values():
    init_x = np.array([1, 1, 1, 1])
    dev = np.diag([1.1, 1.2, 0.9, 1])
    states = [(init_x @ dev ** i).round(2) for i in range(1, 11)]
    return np.array(states)


def test_integration_vb():
    d_t = 1
    F_l = np.array([[1., 0., d_t, 0.],
                    [0., 1., 0., d_t],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    H_k = np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.]])

    m = H_k.shape[0]
    n = H_k.shape[1]

    # these are probably needed for comparison purposes
    # k = 1
    # T = 1000
    # q = 1
    # Q_k_true = get_Q_k(k, T, q, d_t)

    # r = 100
    # R_k_true = get_R_k(k, T, r)

    alpha = 1
    Q_k = alpha * np.eye(4)

    beta = 100
    R_k = beta * np.eye(2)

    tau = 3
    rho = 1 - np.exp(-4)
    N = 10

    ### END of parameter simulation description in the paper

    # let's make some true values of the development
    true_values = get_true_values()

    Q_l = Q_k  # probably?

    # Not sure how to init these...
    P = None
    u = None
    U = None

    # ^ I thought about sampling them somehow, but then I am still
    # missing values needed to sample (P, t, T, u, U,...)

    # iteration of the whole filter
    for state in true_values:
        x_k = state + np.random.rand(4) * 0.1
        z_k = state + np.random.rand(4) * 0.5

        x_k, P, u, U = update(
            x_k, P, u, U, F_l,
            H_k, z_k, Q_l,
            m, n, tau, rho, N
        )
