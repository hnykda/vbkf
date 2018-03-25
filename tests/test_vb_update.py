import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest

from vbkf.vb_update import (get_ts, get_B_i_k, get_us,
                            _get_E_X, get_K_j_k, get_x_j_k__k,
                            get_P_j_k__k, get_A_i_k)
from vbkf import vb_update


def test_get_A():
    P_i_k__k = np.eye(2)
    x_i_k__k = np.array([4, 2])
    x_k__l = np.array([2, 10])
    exp = np.array([[68, 0.],
                    [0., 68]])
    obs = get_A_i_k(P_i_k__k, x_i_k__k, x_k__l)
    aae(exp, obs)


def test_get_ts():
    t_k__l = 2
    A_k = np.eye(2)
    T_k__l = np.eye(2) * 2

    exp_t, exp_T = (3, np.array([
        [3, 0],
        [0, 3]
    ]))
    out_t, out_T = get_ts(t_k__l, A_k, T_k__l)

    aae(exp_T, out_T)
    assert exp_t, out_t


def test_get_B():
    z_k = np.array([2, 3])
    H_k = np.eye(2) * 3
    x_i_k__k = np.array([3, 3])
    P_i_k__k = np.eye(2)
    obs = get_B_i_k(z_k, H_k, x_i_k__k, P_i_k__k)
    exp = np.array([[94., 85.],
                    [85., 94.]])
    aae(exp, obs)


def test_get_us():
    u_k__l = 2
    B_i_k = np.eye(2)
    U_k__l = np.eye(2)
    exp_u, exp_U = 3, np.eye(2) * 2
    obs_u, obs_U = get_us(u_k__l, B_i_k, U_k__l)
    assert exp_u == obs_u
    aae(exp_U, obs_U)


def test_get_E_X():
    w = 2
    n = 3
    W = np.array([[1, 2], [3, 4]])
    obs = _get_E_X(w, n, W)
    exp = np.array([[4., -2.],
                    [-3., 1.]])
    aae(obs, exp)


def test_get_K():
    P_j_k__l = np.array([[1, 2], [3, 4]])
    H_k = np.array([[1, 2], [3, 4]]) * 2
    R_j_k = np.array([[1, 2], [3, 4]]) * 3
    obs = get_K_j_k(P_j_k__l, H_k, R_j_k)
    exp = np.array([[6500, 14468],
                    [14692, 32700]])
    aae(exp, obs)


def test_get_x_j_k_k():
    H_k = np.array([[1, 2], [3, 4]])
    K_j__k = np.array([[1, 2], [3, 4]]) * 2
    z_k = np.array([2, 3])
    x_k__l = np.array([2, 3]) * 2

    obs = get_x_j_k__k(x_k__l, K_j__k, z_k, H_k)
    exp = np.array([-156, -342])
    aae(exp, obs)


def test_get_P_j_k__k():
    P_j_k__l = np.array([[1, 2], [3, 4]])
    K_j_k = np.array([[1, 2], [3, 4]]) * 2
    H_k = np.array([[1, 2], [3, 4]]) * 3
    exp = np.array([[-221, -322],
                    [-483, -704]])
    obs = get_P_j_k__k(P_j_k__l, K_j_k, H_k)
    aae(exp, obs)


def test_single_step():
    x_k__l = np.array([2, 3])
    P = np.diag([1, 2])
    n = 0.3
    m = 0.8
    z_k = np.array([2.5, 2.5])
    H_k = np.diag([3, 2])
    x = x_k__l + np.random.randn(2)
    u = 3
    t = 2
    U = np.array([[1, 2], [3, 4]])
    T = np.array([[1, 2], [3, 4]])
    x, P, t, T, u, U = vb_update.single_step(P, x, x_k__l, t, T, z_k, H_k, u, U, m, n)

    x_ = np.array([-3001.00531972, -5814.21816928])

    P_ = np.array([[-4054.26737401, -5678.12268095],
                   [-7676.08895374, -11515.39641006]])
    t_ = 3
    T_ = np.array([[3.28474663, 2.],
                   [3, 8.56949326]])
    u_ = 4
    U_ = np.array([[90.48182183, 82.48182183],
                   [83.48182183, 92.48182183]])
    aae(x, x_)
    aae(P, P_)
    aae(T, T_)
    aae(U, U_)
    assert t == t_
    assert u == u_

@pytest.mark.skip()
def test_vmu_integration():
    x_k__l = np.array([2, 3])*0.1
    P_k__l = np.diag([1, 2])*0.1
    tau = 0.23
    n = 0.002
    rho = 2
    u_l__l = 0.4
    m = 0.003
    U_l__l = np.array([[1, 2], [3, 4]])*0.1
    z_k = np.array([2.5, 2.5])*0.1
    H_k = np.array([[6, 2], [5, 4]])*0.1
    x, P, t, T, u, U = vb_update.perform_update(x_k__l, P_k__l, tau, n, rho, u_l__l, m, U_l__l, z_k, H_k)
    x_k, P_k__l, R_k = vb_update.sample(x, P, t, T, u, U)
    exp = None
    assert x_k == exp
    assert P_k__l == exp
    assert R_k == exp
