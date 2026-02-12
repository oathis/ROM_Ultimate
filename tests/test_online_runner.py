from rom.runners.static_runner import StaticRunner


def test_static_runner_step():
    runner = StaticRunner()
    runner.load_artifacts("m", "n")
    result = runner.step({"x": 1})
    assert result["mode"] == "m"
