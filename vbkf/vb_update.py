"""
Implementation of Algorithm 1 from
A Novel Adaptive Kalman Filter With Inaccurate Process and Measurement Noise Covariance Matrices
http://ieeexplore.ieee.org/document/8025799/ from 2018.

Notation:
y_k__l := y_{k|l}
l := k-1
y_N_k__l := y^N_{k|l}
lowercased letters: vectors or scalars
uppercased letters: matrices
"""

import numpy as np
from scipy import stats
import logging

DEFAULT_SEED = 2

logger = logging.getLogger(__name__)
np.random.seed(10)


def predict_state(F_l, x_l__l):
    """
    Step 1
    """
    return F_l @ x_l__l


def predict_PECM(F_l, P_l__l, Q_l):
    """
    Step 2
    """
    return F_l @ P_l__l @ F_l.T + Q_l


def time_update(F_l, x_l__l, P_l__l, Q_l):
    x_k__l = predict_state(F_l, x_l__l)
    P_k__l = predict_PECM(F_l, P_l__l, Q_l)
    return x_k__l, P_k__l


def init(P_k__l, tau, n, rho, u_l__l, m, U_l__l):
    """
    Step 3.
    """
    t_k__l = n + tau + 1
    T_k__l = tau * P_k__l
    u_k__l = rho * (u_l__l - m - 1) + m + 1
    U_k__l = rho * U_l__l
    return t_k__l, T_k__l, u_k__l, U_k__l


def get_A_i_k(P_i_k__k, x_i_k__k, x_k__l):
    """
    Step 4

    This is zero for the first iteration...
    """
    x_gain = (x_i_k__k - x_k__l)
    return P_i_k__k * (x_gain @ x_gain.T)


def get_ts(t_k__l, A_i_k, T_k__l):
    """
    Step 5.
    """
    t = t_k__l + 1
    T = A_i_k + T_k__l
    return t, T


def get_B_i_k(z_k, H_k, x_i_k__k, P_i_k__k):
    """
    Step 6
    """
    diff = (z_k - H_k @ x_i_k__k)
    return (diff @ diff.T) + (H_k @ P_i_k__k @ H_k.T)


def get_us(u_k__l, B_i_k, U_k__l):
    """
    Step 7
    """
    u = u_k__l + 1
    U = B_i_k + U_k__l
    return u, U


def _get_E_X(x, n, M):
    return (x - n - 1) * np.linalg.inv(M)


def get_Es(u_j_k, m, U_j_k, t_j_k, n, T_j_k):
    """
    Step 8
    """
    E_R = _get_E_X(u_j_k, m, U_j_k)
    E_P = _get_E_X(t_j_k, n, T_j_k)
    return E_R, E_P


def get_P_R(E_P, E_R):
    """
    Step 9
    """
    return np.linalg.inv(E_P), np.linalg.inv(E_R)


def get_K_j_k(P_j_k__l, H_k, R_j_k):
    """
    Step 10
    """
    return (P_j_k__l @ H_k.T) @ ((H_k @ P_j_k__l @ H_k.T) + R_j_k)


def get_x_j_k__k(x_k__l, K_j_k, z_k, H_k):
    """
    Step 11
    """
    return x_k__l + (K_j_k @ (z_k - H_k @ x_k__l))


def get_P_j_k__k(P_j_k__l, K_j_k, H_k):
    """
    Step 12
    """
    return P_j_k__l - (K_j_k @ H_k @ P_j_k__l)


def single_step(P_i_k__k, x_i_k__k, x_k__l, t_k__l, T_k__l, z_k, H_k, u_k__l, U_k__l, m, n):
    """
    A single step in the for loop. Basically applies steps 4.-12.
    """
    A_i_k = get_A_i_k(P_i_k__k, x_i_k__k, x_k__l)
    t_j_k, T_j_k = get_ts(t_k__l, A_i_k, T_k__l)
    B_i_k = get_B_i_k(z_k, H_k, x_i_k__k, P_i_k__k)
    u_j_k, U_j_k = get_us(u_k__l, B_i_k, U_k__l)
    E_R, E_P = get_Es(u_j_k, m, U_j_k, t_j_k, n, T_j_k)
    P_j_k__l, R_j_k = get_P_R(E_P, E_R)
    K_j_k = get_K_j_k(P_j_k__l, H_k, R_j_k)
    x_j_k_k = get_x_j_k__k(x_k__l, K_j_k, z_k, H_k)
    P_j_k__k = get_P_j_k__k(P_j_k__l, K_j_k, H_k)
    return x_j_k_k, P_j_k__k, t_j_k, T_j_k, u_j_k, U_j_k


def perform_update(x_k__l, P_k__l, tau, n, rho, u_l__l, m, U_l__l, z_k, H_k, N):
    """
    Variational measurement update. Initialize variables (step 3.),
    performs the loop (steps 4.-12.) and returns updated values (step 13.).
    """
    t, T, u, U = init(P_k__l, tau, n, rho, u_l__l, m, U_l__l)
    P = P_k__l
    x = x_k__l
    for i in range(N):
        x, P, t, T, u, U = single_step(P, x, x_k__l, t, T, z_k, H_k, u, U, m, n)
        logger.debug('[VB step %s]', i)
    return x, P, t, T, u, U


def sample(x_k__k, P_k__k, t_k__k, T_k__k, u_k__k, U_k__k):
    """
    Actual sampling from q(x_k), q(P_k__l), q(R_k) based on the values
    from the Algorithm 1.
    """
    x_k = stats.norm.rvs(x_k__k, P_k__k, random_state=DEFAULT_SEED)
    P_k__l = stats.invwishart.rvs(t_k__k, T_k__k, random_state=DEFAULT_SEED)
    R_k = stats.invwishart.rvs(u_k__k, U_k__k, random_state=DEFAULT_SEED)
    return x_k, P_k__l, R_k


def variational_measurement_update(x_k__l, P_k__l, tau, n, rho, u_l__l, m, U_l__l, z_k, H_k, N):
    """
    Wrapper to perform the Variational measurement update and
    then sample based on the results (eq. 46-48).
    """
    x, P, t, T, u, U = perform_update(x_k__l, P_k__l, tau, n, rho, u_l__l, m, U_l__l, z_k, H_k, N)
    x_k, P_k__l, R_k = sample(x, P, t, T, u, U)
    return x_k, P_k__l, u, U


def update(x_l__l, P_l__l, u_l__l, U_l__l, F_l, H_k, z_k, Q_l, m, n, tau, rho, N):
    """
    Functionality of Algorithm 1 AND sampling (eq. 46-48).
    """
    logger.debug('X_init %s', x_l__l)
    x_k__l, P_k__l = time_update(F_l, x_l__l, P_l__l, Q_l)
    logger.debug('X_predicted %s', x_k__l)
    x_k, P_k__l, u, U = variational_measurement_update(
        x_k__l, P_k__l, tau, n, rho, u_l__l, m, U_l__l, z_k, H_k, N
    )
    logger.debug('X_updated %s', x_k)
    return x_k, P_k__l, u, U
