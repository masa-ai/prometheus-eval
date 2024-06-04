from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_prometheus2_config(
    yaml_path: Path = Path.cwd()
    / "libs"
    / "prometheus-eval"
    / "libs"
    / "prometheus-eval"
    / "prometheus_eval"
    / "configs.yaml",
) -> DictConfig | ListConfig:
    return OmegaConf.load(yaml_path)
