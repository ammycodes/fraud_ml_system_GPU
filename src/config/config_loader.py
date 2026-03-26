
import yaml

class ConfigLoader:
    """Loads and validates config"""

    def __init__(self, path):
        self.path = path

    def load(self):
        with open(self.path) as f:
            cfg = yaml.safe_load(f)

        self._validate(cfg)
        return cfg

    def _validate(self, cfg):
        assert "data" in cfg, "Missing data config"
        assert "target" in cfg["data"], "Target not defined"
