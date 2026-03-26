
from sklearn.model_selection import train_test_split

class Splitter:

    def __init__(self, cfg):
        self.cfg = cfg

    def split(self, df):

        target = self.cfg["data"]["target"]
        ratios = self.cfg["split"]["ratios"]

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1 - ratios["train"]),
            stratify=y,
            random_state=self.cfg["split"]["random_seed"]
        )

        return X_train, X_temp, y_train, y_temp
