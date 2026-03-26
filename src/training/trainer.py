
import pandas as pd
from tqdm import tqdm
from src.models.factory import get_models
from src.evaluation.metrics import compute_metrics
from src.training.model_saver import save_model
from src.training.tuner import Tuner
from src.training.threshold import ThresholdOptimizer
from src.training.calibrator import Calibrator
from src.data.sampler import Sampler

from src.explainability.shap_engine import ShapEngine
from src.explainability.importance import FeatureImportance
from src.ensemble.blender import Blender
from src.ensemble.selector import ModelSelector
from src.explainability.report import ReportGenerator

class Trainer:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(self, df):

        if "polars" in str(type(df)):
            df = df.to_pandas()

        y = df[self.cfg["data"]["target"]]
        X = df.drop(columns=[self.cfg["data"]["target"]])

        pos = sum(y)
        neg = len(y) - pos
        w = neg / max(pos, 1)

        sampler = Sampler(self.cfg, self.logger)
        X, y = sampler.run(X, y)

        tuner = Tuner(self.cfg, self.logger)
        threshold_opt = ThresholdOptimizer(self.cfg)

        shap_engine = ShapEngine(self.cfg, self.logger)
        fi_engine = FeatureImportance()

        models = get_models(self.cfg)

        results = []
        preds = []

        for m in tqdm(models):

            try:
                self.logger.info(f"Training {m.__class__.__name__}")

                m = tuner.tune(m, X, y)
                m.fit(X, y, w)

                p = m.predict(X)

                if self.cfg.get("calibration", {}).get("enabled", False):
                    cal = Calibrator()
                    m = cal.run(m, X, y)
                    p = m.predict(X)

                metrics = compute_metrics(y, p)
                t = threshold_opt.find(y.values, p)

                metrics["threshold"] = t
                metrics["model"] = m.__class__.__name__

                # SHAP
                shap_df = shap_engine.run(m, X)
                if shap_df is not None:
                    shap_df.to_csv(f"outputs/shap/{m.__class__.__name__}.csv")

                # Feature importance
                fi = fi_engine.run(m, X)
                if fi is not None:
                    fi.to_csv(f"outputs/feature_importance/{m.__class__.__name__}.csv")

                save_model(m, f"outputs/models/{m.__class__.__name__}", metrics)

                results.append(metrics)
                preds.append(p)

            except Exception as e:
                self.logger.error(f"{m} failed: {e}")
                continue

        dfm = pd.DataFrame(results)
        dfm.to_csv("outputs/metrics.csv", index=False)

        # Ensemble
        selector = ModelSelector()
        top_models = selector.select(dfm)

        blender = Blender()
        final_pred = blender.blend(preds[:len(top_models)], top_models["auc_pr"].values)

        pd.DataFrame({"pred": final_pred}).to_csv("outputs/predictions/ensemble.csv", index=False)

        # Report
        ReportGenerator().generate(dfm, "outputs/model_explanations.md")
