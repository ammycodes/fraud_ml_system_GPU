
import pandas as pd

class ShapEngine:
    """Computes SHAP values (sampled for performance)"""

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(self, model, X):

        if not self.cfg["explainability"]["shap"]["enabled"]:
            return None

        try:
            import shap

            sample_size = self.cfg["explainability"]["shap"]["sample_size"]
            X_sample = X.sample(min(len(X), sample_size))

            explainer = shap.Explainer(model.model, X_sample)
            shap_values = explainer(X_sample)

            df = pd.DataFrame(shap_values.values).mean().to_frame("importance")
            return df

        except Exception as e:
            self.logger.warning(f"SHAP failed: {e}")
            return None
