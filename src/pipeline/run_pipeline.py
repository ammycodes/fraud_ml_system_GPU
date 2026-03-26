
from src.config.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.data.loader import Loader
from src.training.trainer import Trainer
from src.inference.infer_engine import InferenceEngine
from src.observability.run_summary import save_summary

def run():

    cfg = ConfigLoader("config/config.yaml").load()
    logger = get_logger()

    df = Loader(cfg, logger).load()

    Trainer(cfg, logger).run(df)

    InferenceEngine(cfg, logger).run()

    save_summary()

if __name__ == "__main__":
    run()
