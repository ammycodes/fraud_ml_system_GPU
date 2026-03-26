
"""Command Line Interface for Fraud ML System"""

import argparse
from src.pipeline.run_pipeline import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    run()

if __name__ == "__main__":
    main()
