
from sklearn.calibration import CalibratedClassifierCV

class Calibrator:
    """Probability calibration"""

    def __init__(self, method="sigmoid"):
        self.method = method

    def run(self, model, X, y):

        try:
            cal = CalibratedClassifierCV(model.model, method=self.method)
            cal.fit(X, y)
            model.model = cal
        except:
            pass

        return model
