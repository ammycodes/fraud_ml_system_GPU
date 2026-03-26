
import os, joblib, json

def save_model(model, path, metrics):

    os.makedirs(path, exist_ok=True)

    joblib.dump(model, f"{path}/model.pkl")

    with open(f"{path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
