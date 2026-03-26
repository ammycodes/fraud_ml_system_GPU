
import numpy as np

class ThresholdOptimizer:
    """Find optimal threshold for fraud use-case"""

    def __init__(self, cfg):
        self.cfg = cfg

    def find(self, y, p):

        thresholds = np.linspace(0.01, 0.99, 50)

        best_t = 0.5
        best_score = -1

        for t in thresholds:

            pred = (p > t).astype(int)

            precision = (pred & y).sum() / max(pred.sum(), 1)
            recall = (pred & y).sum() / max(y.sum(), 1)

            if precision >= 0.9 and recall > best_score:
                best_score = recall
                best_t = t

        return best_t
