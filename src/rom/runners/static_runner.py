from rom.interfaces.online_runner import OnlineRunner


class StaticRunner(OnlineRunner):
    def load_artifacts(self, mode_path: str, model_path: str):
        self._mode_path = mode_path
        self._model_path = model_path

    def step(self, online_input):
        return {"input": online_input, "mode": self._mode_path, "model": self._model_path}
