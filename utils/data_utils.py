"""
Data Loading Utility functions
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from generate_data import generate_lsn


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node


class SyntheticDataset(Dataset):

    def __init__(self, X, y, adj, tau=None, gamma=None, noise=None, drop_rate=0):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.adj = adj

        if noise == "drop":
            self._random_drop(drop_rate)
        elif noise == "flip":
            self._random_flip(drop_rate)

        self.sigma = self._get_covariance(tau, gamma)
        self.edge_index = torch.tensor(np.array(list(self.adj.nonzero())))

        n = self.__len__()
        self.train_mask = torch.zeros(n).to(dtype=torch.bool)
        self.train_mask[:config("train_size")] = True
        self.val_mask = torch.zeros(n).to(dtype=torch.bool)
        self.val_mask[config("train_size"):config("train_size") + config("valid_size")] = True
        self.test_mask = torch.zeros(n).to(dtype=torch.bool)
        self.test_mask[-config("test_size"):] = True

        self.sigma_inv_train = self.sigma[self.train_mask].t()[self.train_mask].t()
        self.sigma_inv_train = self.sigma_inv_train.inverse()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _random_drop(self, rate):
        for i in range(0, self.__len__()):
            nonzero = np.where(self.adj[i] != 0)[0]
            for idx in nonzero:
                if idx == i:
                    continue
                drop = np.random.choice([True, False], 1, p=[rate, 1-rate])[0]
                if drop:
                    self.adj[i][idx] = 0

    def _random_flip(self, rate):
        for i in range(0, self.__len__()):
            nonzero = np.where(self.adj[i] != 0)[0]
            for idx in nonzero:
                if idx == i:
                    continue
                drop = np.random.choice([True, False], 1, p=[rate, 1-rate])[0]
                if drop:
                    self.adj[i][idx] = 0
                    x = np.random.randint(0, self.__len__())
                    y = np.random.randint(0, self.__len__())
                    self.adj[x][y] = 1

    def _get_covariance(self, tau, gamma):
        if not tau or not gamma:
            return None
        L = np.diag(self.adj.sum(axis=0)) - self.adj
        cov = tau * np.linalg.inv(L + gamma * np.eye(self.__len__()))
        return torch.tensor(cov)


def get_dataset(n=300,
                d0=10,
                d1=10,
                m=3000,
                gamma=0.05,
                tau=0.1,
                seed=0,
                mean_mode="xw",
                cov_mode="li",
                noise=None,
                drop_rate=0):

    X, y, adj, _ = generate_lsn(n=n,
                                d0=d0,
                                d1=d1,
                                m=m,
                                gamma=gamma,
                                tau=tau,
                                seed=seed,
                                mean_mode=mean_mode,
                                cov_mode=cov_mode)
    return SyntheticDataset(X, y, adj,
                            tau=tau,
                            gamma=gamma,
                            noise=noise,
                            drop_rate=drop_rate)
