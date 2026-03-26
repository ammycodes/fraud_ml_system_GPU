
import polars as pl

class Batcher:
    """Handles batch-wise reading using Polars"""

    def __init__(self, cfg):
        self.cfg = cfg

    def read_batches(self, path):

        batch_size = self.cfg["inference"]["batching"]["batch_size"]

        df = pl.scan_parquet(path)

        total = df.select(pl.count()).collect()[0,0]

        for i in range(0, total, batch_size):
            yield df.slice(i, batch_size).collect()
