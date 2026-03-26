
class BaseModel:
    """Base model with unified interface"""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.model = None

    def fit(self, X, y, weight):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
