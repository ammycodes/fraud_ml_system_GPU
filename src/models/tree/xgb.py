
import xgboost as xgb
from src.models.base_model import BaseModel

class XGBModel(BaseModel):

    def __init__(self, cfg):
        super().__init__("xgb", cfg)

    def fit(self, X, y, w):
        params = self.cfg.get("params", {})

        try:
            params["tree_method"] = "gpu_hist"
            params["scale_pos_weight"] = w
            self.model = xgb.XGBClassifier(**params)
        except:
            self.model = xgb.XGBClassifier(**params)

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
