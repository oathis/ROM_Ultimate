from rom.interfaces.mode_builder import ModeBuilder


class DMDBuilder(ModeBuilder):
    def fit(self, snapshots):
        self._snapshots = snapshots
        return self

    def transform(self, snapshots):
        return snapshots

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("DMDBuilder placeholder artifact")
