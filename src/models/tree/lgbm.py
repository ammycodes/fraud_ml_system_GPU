
import lightgbm as lgb
from src.models.base_model import BaseModel

class LGBMModel(BaseModel):

    def __init__(self, cfg):
        super().__init__("lgbm", cfg)

    def fit(self, X, y, w):
        params = self.cfg.get("params", {})

        try:
            params["device"] = "gpu"
            params["scale_pos_weight"] = w
            self.model = lgb.LGBMClassifier(**params)
        except:
            self.model = lgb.LGBMClassifier(**params)

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
