
import glob

class Loader:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def load(self):

        path = self.cfg["data"]["train_path"]

        try:
            import dask_cudf
            self.logger.info("Using Dask-cuDF")
            return dask_cudf.read_parquet(path)
        except:
            pass

        try:
            import cudf
            self.logger.info("Using cuDF")
            files = glob.glob(f"{path}/*.parquet")
            return cudf.concat([cudf.read_parquet(f) for f in files])
        except:
            pass

        try:
            import dask.dataframe as dd
            self.logger.info("Using Dask CPU")
            return dd.read_parquet(path)
        except:
            pass

        import pandas as pd
        self.logger.info("Using Pandas fallback")
        return pd.read_parquet(path)
