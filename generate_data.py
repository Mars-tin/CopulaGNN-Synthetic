"""
CovGNN/generate_data.py.

Generate synthetic data in data/lsn/.
"""
import os
import numpy as np
import pickle


def generate_lsn(n=300,
                 d0=10,
                 d1=10,
                 m=3000,
                 gamma=0.05,
                 tau=0.1,
                 seed=0,
                 mean_mode="xw",
                 cov_mode="li",
                 root='./data',
                 save_file=False):

    path = os.path.join(root, "lsn")
    assert mean_mode in ["xw", "daxw"]
    assert cov_mode in ["li", "i"]

    filename = "lsn_{}_{}_n{}_d0{}_d1{}_m{}_g{}_t{}_s{}.pkl".format(
        mean_mode, cov_mode, n, d0, d1, m, gamma, tau, seed)

    if not save_file and os.path.exists(os.path.join(path, filename)):
        data, params = pickle.load(open(os.path.join(path, filename), "rb"))
        return data[0], data[1], data[2], filename

    rs = np.random.RandomState(seed=seed)

    x = rs.normal(size=(n, d0))
    w_a = rs.normal(size=(d0, d1))
    w_y = rs.normal(size=(d0, ))

    prod = x.dot(w_a)  # (n, d1)
    logits = -np.linalg.norm(
        prod.reshape(1, n, d1) - prod.reshape(n, 1, d1), axis=2
    )  # (n, n)
    threshold = np.sort(logits.reshape(-1))[-m]
    adj = (logits >= threshold).astype(float)
    L = np.diag(adj.sum(axis=0)) - adj

    if mean_mode == "xw":
        y_mean = x.dot(w_y)
    else:
        y_mean = np.diag(1. / adj.sum(axis=0)).dot(adj).dot(x).dot(w_y)

    if cov_mode == "li":
        y_cov = tau * np.linalg.inv(L + gamma * np.eye(n))
    else:
        y_cov = np.eye(n)

    y = rs.multivariate_normal(y_mean, y_cov)

    if save_file:
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(((x, y, adj), (w_a, w_y)),
                    open(os.path.join(path, filename), "wb"))

    return x, y, adj, filename


if __name__ == "__main__":
    _d0 = 10
    _d1 = 10
    _n = 300
    _m = [10*_n, 20*_n, 30*_n]
    _gamma = [0.05, 0.5, 5]
    _tau = [0.1, 1, 10]
    _seed = range(20)

    print("Start generating synthetic data.\n"
          "This may take a few minutes...")

    for seed_val in _seed:
        for m_val in _m:
            for gamma_val in _gamma:
                for tau_val in _tau:
                    generate_lsn(
                        seed=seed_val,
                        m=m_val,
                        gamma=gamma_val,
                        tau=tau_val,
                        mean_mode="xw",
                        cov_mode="li",
                        save_file=True)

                    generate_lsn(
                        seed=seed_val,
                        m=m_val,
                        gamma=gamma_val,
                        tau=tau_val,
                        mean_mode="daxw",
                        cov_mode="i",
                        save_file=True)

                    generate_lsn(
                        seed=seed_val,
                        m=m_val,
                        gamma=gamma_val,
                        tau=tau_val,
                        mean_mode="daxw",
                        cov_mode="li",
                        save_file=True)

    print("Successfully Generated synthetic data!")
