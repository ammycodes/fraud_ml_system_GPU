
import optuna
import numpy as np
from src.evaluation.metrics import compute_metrics

class Tuner:
    """Hyperparameter tuning using Optuna (Ray scaffold ready)"""

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def tune(self, model, X, y):

        if not self.cfg.get("tuning", {}).get("enabled", False):
            return model

        def objective(trial):

            # Example search space (extend per model)
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3)
            }

            try:
                model.cfg["params"].update(params)
            except:
                pass

            model.fit(X, y, weight=1)

            p = model.predict(X)
            score = compute_metrics(y, p)["auc_pr"]

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.cfg["tuning"].get("n_trials", 10))

        self.logger.info(f"Best params: {study.best_params}")

        return model
