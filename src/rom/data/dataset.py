from dataclasses import dataclass


@dataclass
class DatasetBundle:
    snapshots: list
    params: list


def load_dataset(_cfg) -> DatasetBundle:
    return DatasetBundle(snapshots=[], params=[])
