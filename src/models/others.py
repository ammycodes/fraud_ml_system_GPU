
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

class SKRF:
    def fit(self, X, y, w):
        self.m = RandomForestClassifier(n_jobs=-1)
        self.m.fit(X, y)
    def predict(self, X): return self.m.predict_proba(X)[:,1]

class ExtraTrees:
    def fit(self, X, y, w):
        self.m = ExtraTreesClassifier(n_jobs=-1)
        self.m.fit(X, y)
    def predict(self, X): return self.m.predict_proba(X)[:,1]

class HistGB:
    def fit(self, X, y, w):
        self.m = HistGradientBoostingClassifier()
        self.m.fit(X, y)
    def predict(self, X): return self.m.predict_proba(X)[:,1]

class Logistic:
    def fit(self, X, y, w):
        self.m = LogisticRegression(max_iter=1000)
        self.m.fit(X, y)
    def predict(self, X): return self.m.predict_proba(X)[:,1]

class IsoForest:
    def fit(self, X, y, w):
        self.m = IsolationForest()
        self.m.fit(X)
    def predict(self, X): return self.m.decision_function(X)
