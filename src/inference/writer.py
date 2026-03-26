
import os

def write_output(df, path, idx):

    os.makedirs(path, exist_ok=True)
    df.write_parquet(f"{path}/pred_{idx}.parquet")
