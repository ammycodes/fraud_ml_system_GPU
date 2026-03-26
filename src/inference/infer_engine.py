
import multiprocessing as mp
import pandas as pd

from src.inference.model_loader import load_models
from src.inference.batcher import Batcher
from src.inference.writer import write_output

class InferenceEngine:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def _score_batch(self, batch, models):

        pdf = batch.to_pandas()

        preds = []

        for name, m in models.items():
            try:
                preds.append(m.predict_proba(pdf)[:,1])
            except:
                preds.append(m.predict(pdf))

        final = sum(preds) / len(preds)

        return pd.DataFrame({"prediction": final})

    def run(self):

        if not self.cfg["inference"]["enabled"]:
            return

        path = self.cfg["data"]["test_path"]

        models = load_models()

        batcher = Batcher(self.cfg)

        pool = mp.Pool(self.cfg["inference"]["cpu_cores"])

        results = []

        for i, batch in enumerate(batcher.read_batches(path)):

            self.logger.info(f"Scoring batch {i}")

            results.append(
                pool.apply_async(self._score_batch, args=(batch, models))
            )

        for i, r in enumerate(results):
            df = r.get()
            write_output(df, self.cfg["inference"]["output"]["path"], i)

        pool.close()
        pool.join()
