from rom.core.pipeline import run_pipeline
from rom.modes.pod_builder import PODBuilder
from rom.trainers.rbf_trainer import RBFTrainer


def test_pipeline_smoke():
    output = run_pipeline({}, PODBuilder(), RBFTrainer())
    assert "latent" in output
