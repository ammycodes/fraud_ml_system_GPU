
import mlflow

class MLflowManager:

    def __init__(self, cfg):
        self.cfg = cfg

    def start(self):
        mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
        return mlflow.start_run()

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, k, v):
        mlflow.log_metric(k, v)
