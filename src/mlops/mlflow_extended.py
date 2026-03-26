
import mlflow

class MLflowExtended:

    def __init__(self, cfg):
        self.cfg = cfg

    def log_model(self, name, metrics):
        for k, v in metrics.items():
            mlflow.log_metric(f"{name}_{k}", v)

    def log_artifact(self, path):
        mlflow.log_artifact(path)
