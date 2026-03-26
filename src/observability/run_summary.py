
import json
import os
from datetime import datetime

def save_summary(metrics_path="outputs/metrics.csv"):

    summary = {
        "timestamp": str(datetime.now()),
        "metrics_file": metrics_path
    }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
