from rom.data.dataset import load_dataset


def run_pipeline(cfg, mode_builder, trainer):
    data = load_dataset(cfg)
    mode_builder.fit(data.snapshots)
    latent = mode_builder.transform(data.snapshots)
    trainer.fit(data.params, latent)
    return {"latent": latent}
