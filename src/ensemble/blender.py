
import numpy as np

class Blender:
    """Weighted ensemble based on AUC-PR"""

    def blend(self, preds, scores):

        weights = np.array(scores)
        weights = weights / weights.sum()

        final = np.zeros_like(preds[0])

        for w, p in zip(weights, preds):
            final += w * p

        return final
