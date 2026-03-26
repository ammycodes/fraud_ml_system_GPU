
import pandas as pd

class FeatureImportance:

    def run(self, model, X):

        try:
            imp = model.model.feature_importances_
            return pd.DataFrame({
                "feature": X.columns,
                "importance": imp
            }).sort_values("importance", ascending=False)
        except:
            return None
