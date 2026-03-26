
import time

class ResourceManager:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def wait_for_gpu(self):

        retry = self.cfg["resources"]["gpu"]["retry"]

        for i in range(retry["max_retries"]):
            try:
                import torch
                if torch.cuda.is_available():
                    return True
            except:
                pass

            self.logger.warning(f"GPU not available, retry {i}")
            time.sleep(retry["wait_seconds"])

        self.logger.warning("Falling back to CPU")
        return False
