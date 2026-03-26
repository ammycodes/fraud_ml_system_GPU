
import joblib
import os

def load_models(model_dir="outputs/models"):
    models = {}

    for m in os.listdir(model_dir):
        path = f"{model_dir}/{m}/model.pkl"
        if os.path.exists(path):
            models[m] = joblib.load(path)

    return models
