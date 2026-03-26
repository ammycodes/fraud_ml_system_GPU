
import pandas as pd

class ModelSelector:
    """Select top-K models"""

    def select(self, metrics_df, k=3):
        return metrics_df.sort_values("auc_pr", ascending=False).head(k)
