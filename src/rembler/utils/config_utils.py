from __future__ import annotations

from typing import Any

import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

BASE_DIR = "/path/to/your/base/dir"  # Update this path as needed


def load_config_from_hydra_dump(experiment_name: str, base_dir: str = BASE_DIR) -> dict[str, Any]:
    """
    Load a configuration from a Hydra dump file.

    Args:
        experiment_name (str): The name of the experiment.
        base_dir (str, optional): The base directory where the dump file is located. Defaults to BASE_DIR.

    Returns:
        dict: The loaded configuration.

    """
    filename = f"{base_dir}/{experiment_name}/.hydra/config.yaml"
    with open(filename) as file:
        config = yaml.safe_load(file)
    return config


def load_config_with_compose_api(config_path: str = "config", overrides: list[str] | None = None) -> dict[str, Any]:
    """
    Load a configuration file using the Hydra Compose API.

    Args:
        config_path (str, optional): The path to the configuration file. Defaults to "config".
        overrides (list, optional): A list of overrides to apply to the configuration. Defaults to [].

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    if overrides is None:
        overrides = []
    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        cfg = compose(config_name="config", overrides=overrides)
        return yaml.safe_load(OmegaConf.to_yaml(cfg))


def unpack_config(config: dict[str, Any]) -> tuple[None, None, None, None, None]:
    """
    Unpacks the configuration dictionary into individual specification objects.

    Parameters:
        config (dict): The configuration dictionary containing specs.

    Returns:
        tuple: A tuple containing the unpacked spec objects.
    """
    return None, None, None, None, None
