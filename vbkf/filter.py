import numpy as np
import logging

logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)


def get_stupid_model(dim):
    return np.eye(dim) * 1.1


def get_stupid_precision(dim):
    # tenths all dimensions
    return np.eye(dim) * .1


def get_stupid_XNCM(dim):
    # well...
    return np.eye(dim)


class KF:
    iterations = None

    true_values = []
    measurements = []
    predictions = []
    corrections = []

    def __init__(self, iterations):
        self.iterations = iterations

    def predict_state(self, F_k, x_l__l):
        return F_k @ x_l__l

    def predict_cov(self, F_k, P_l__l, Q_k):
        return F_k @ P_l__l @ F_k.T + Q_k

    def prediction_step(self, F_k, x_l__l, P_l__l, Q_k):
        x_k__l = self.predict_state(F_k, x_l__l)
        P_k__l = self.predict_cov(F_k, P_l__l, Q_k)
        return x_k__l, P_k__l

    def update_innov_residual(self, H_k, z_k, x_k__l):
        return z_k - H_k @ x_k__l

    def update_innov_cov(self, H_k, R_k, P_k__l):
        return R_k + H_k @ P_k__l @ H_k.T

    def update_innov(self, H_k, z_k, x_k__l, R_k, P_k__l):
        y_k = self.update_innov_residual(H_k, z_k, x_k__l)
        S_k = self.update_innov_cov(H_k, R_k, P_k__l)
        return y_k, S_k

    def update_kalman_gain(self, P_k__l, H_k, S_k):
        return P_k__l @ H_k.T @ np.linalg.inv(S_k)

    def update_aposteriori_state(self, x_k__l, K_k, y_k):
        return x_k__l + K_k @ y_k

    def update_aposteriori_cov(self, K_k, H_k, P_k__l, R_k):
        D = np.eye(K_k.shape[0]) - K_k @ H_k
        return (D @ P_k__l @ D.T) + (K_k @ R_k @ K_k.T)

    def update_postfit_residual(self, z_k, H_k, x_k__k):
        return z_k - H_k @ x_k__k

    def update_step(self, H_k, R_k, z_k, x_k__l, P_k__l):
        y_k, S_k = self.update_innov(H_k, z_k, x_k__l, R_k, P_k__l)
        K_k = self.update_kalman_gain(P_k__l, H_k, S_k)
        x_k__k = self.update_aposteriori_state(x_k__l, K_k, y_k)
        P_k__k = self.update_aposteriori_cov(K_k, H_k, P_k__l, R_k)
        # y_k__k = self.update_postfit_residual(z_k, H_k, x_k__k)
        return x_k__k, P_k__k

    def perform_stupid_measurement(self, x_k__l):
        """
        Doesn't/shouldn't take X_k__l of course,
        just for demonstration purposes
        """
        true_val = x_k__l + np.random.randn(2)
        self.true_values.append(true_val)
        measure = true_val + np.random.randn(2)
        self.measurements.append(measure)
        return measure

    def evolve_H(self, H):
        return H

    def evolve_R(self, R):
        return R

    def evolve_F(self, F):
        return F

    def evolve_Q(self, Q):
        return Q

    def perform_step_k(self, F_k, Q_k, H_k, R_k, x_l__l, P_l__l):
        """
        Where the k signify k-th prediction step. There can be actually
        multiple predictions done.
        """
        x_k__l, P_k__l = self.prediction_step(F_k, x_l__l, P_l__l, Q_k)
        logger.info(f'Prediction x_k__l={x_k__l}')
        z_k = self.perform_stupid_measurement(x_k__l)
        x_k__k, P_k__k = self.update_step(H_k, R_k, z_k, x_k__l, P_k__l)
        logger.info(f'Corrected x_k__k={x_k__k}')
        self.corrections.append(x_k__k)
        return x_k__k, P_k__k

    def evolve(self, H_l, R_l, F_l, Q_l):
        H_k = self.evolve_H(H_l)
        R_k = self.evolve_R(R_l)
        F_k = self.evolve_F(F_l)
        Q_k = self.evolve_Q(Q_l)
        return H_k, R_k, F_k, Q_k

    def step(self, H_l, R_l, F_l, Q_l, x_l__l, P_l__l):
        H_k, R_k, F_k, Q_k = self.evolve(H_l, R_l, F_l, Q_l)
        logger.info(f'Prediction X={x_l__l}')
        x_k__k, P_k__k = self.perform_step_k(F_k, Q_k, H_k, R_k, x_l__l, P_l__l)
        return H_k, R_k, F_k, Q_k, x_k__k, P_k__k

    def loop(self, H_k, R_k, F_k, Q_k, x_k__k, P_k__k):
        for i in range(self.iterations):
            H_k, R_k, F_k, Q_k, x_k__k, P_k__k = self.step(H_k, R_k, F_k, Q_k, x_k__k, P_k__k)
            logger.info(f'Step {i}')


def init_all(state_dimension):
    x_l__l = np.ones(state_dimension) * 2  # initial state
    F_l = get_stupid_model(state_dimension)  # model

    P_l__l = get_stupid_precision(state_dimension)  # initial prediction cov
    Q_l = get_stupid_XNCM(state_dimension)  # initial prediction cov

    H_l = get_stupid_model(state_dimension)  # observation matrix
    R_l = get_stupid_XNCM(state_dimension)  # initial prediction cov

    return H_l, R_l, F_l, Q_l, x_l__l, P_l__l


import pandas as pd

def unp(x):
    return [i[0] for i in x]

def main():
    # dimension of the state vector, e.g. x = [x_y, v_y]
    state_dimension = 2
    iterations = 40

    # these are more like `l` index, but this make it easy in the loop
    H_k, R_k, F_k, Q_k, x_k__k, P_k__k = init_all(state_dimension)

    kf = KF(iterations)
    kf.loop(H_k, R_k, F_k, Q_k, x_k__k, P_k__k)
    df = pd.DataFrame.from_dict({
        'true_values': unp(kf.true_values),
        'measurements': unp(kf.measurements),
        'corrections': unp(kf.corrections)
    })
    df.to_csv('vals.csv')


main()
