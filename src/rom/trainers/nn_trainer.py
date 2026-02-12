from rom.interfaces.offline_trainer import OfflineTrainer


class NNTrainer(OfflineTrainer):
    def fit(self, x_train, y_train):
        self._shape = (len(x_train), len(y_train))
        return self

    def predict(self, x):
        return x

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("NNTrainer placeholder artifact")
