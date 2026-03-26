
import numpy as np

class Sampler:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(self, X, y):

        if not self.cfg["imbalance"]["enabled"]:
            return X, y

        self.logger.info("Applying hybrid sampling")

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        n_neg = int(len(neg_idx) * 0.1)

        neg_sample = np.random.choice(neg_idx, n_neg, replace=False)
        idx = np.concatenate([pos_idx, neg_sample])

        return X.iloc[idx], y.iloc[idx]
