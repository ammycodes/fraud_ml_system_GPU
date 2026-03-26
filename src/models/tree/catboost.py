
from catboost import CatBoostClassifier
from src.models.base_model import BaseModel

class CatBoostModel(BaseModel):

    def __init__(self, cfg):
        super().__init__("catboost", cfg)

    def fit(self, X, y, w):
        params = self.cfg.get("params", {})

        try:
            params["task_type"] = "GPU"
            params["class_weights"] = [1, w]
        except:
            pass

        self.model = CatBoostClassifier(**params, verbose=0)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
