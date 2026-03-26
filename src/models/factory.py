
from src.models.tree.xgb import XGBModel
from src.models.tree.lgbm import LGBMModel
from src.models.tree.catboost import CatBoostModel
from src.models.others import *

def get_models(cfg):
    models = []

    if cfg["models"].get("xgb", {}).get("enabled"):
        models.append(XGBModel(cfg["models"]["xgb"]))

    if cfg["models"].get("lgbm", {}).get("enabled"):
        models.append(LGBMModel(cfg["models"]["lgbm"]))

    if cfg["models"].get("catboost", {}).get("enabled"):
        models.append(CatBoostModel(cfg["models"]["catboost"]))

    if cfg["models"].get("sklearn_rf", {}).get("enabled"):
        models.append(SKRF())

    if cfg["models"].get("extra_trees", {}).get("enabled"):
        models.append(ExtraTrees())

    if cfg["models"].get("histgb", {}).get("enabled"):
        models.append(HistGB())

    if cfg["models"].get("logistic", {}).get("enabled"):
        models.append(Logistic())

    if cfg["models"].get("isolation_forest", {}).get("enabled"):
        models.append(IsoForest())

    return models
