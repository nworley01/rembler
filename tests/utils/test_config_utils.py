import os
import tempfile
import yaml
from unittest.mock import patch, mock_open, MagicMock
import pytest

from rembler.utils.config_utils import (
    load_config_from_hydra_dump,
    load_config_with_compose_api,
    unpack_config,
    BASE_DIR
)


@pytest.mark.unit
def test_load_config_from_hydra_dump_success():
    """Test successful loading of config from Hydra dump file."""
    test_config = {"model": {"type": "cnn"}, "data": {"batch_size": 32}}
    mock_yaml_content = yaml.dump(test_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        result = load_config_from_hydra_dump("test_experiment", "/tmp/base")

    assert result == test_config


@pytest.mark.unit
def test_load_config_from_hydra_dump_default_base_dir():
    """Test loading config with default base directory."""
    test_config = {"test": "value"}
    mock_yaml_content = yaml.dump(test_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        result = load_config_from_hydra_dump("experiment")

    assert result == test_config


@pytest.mark.unit
def test_load_config_from_hydra_dump_file_not_found():
    """Test handling of missing config file."""
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            load_config_from_hydra_dump("nonexistent", "/tmp/base")


@pytest.mark.unit
def test_load_config_from_hydra_dump_invalid_yaml():
    """Test handling of invalid YAML content."""
    with patch("builtins.open", mock_open(read_data="invalid: yaml: content: [")):
        with pytest.raises(yaml.YAMLError):
            load_config_from_hydra_dump("test_experiment", "/tmp/base")


@pytest.mark.unit
def test_load_config_from_hydra_dump_empty_file():
    """Test handling of empty config file."""
    with patch("builtins.open", mock_open(read_data="")):
        result = load_config_from_hydra_dump("test_experiment", "/tmp/base")

    assert result is None


@pytest.mark.unit
@patch('rembler.utils.config_utils.initialize')
@patch('rembler.utils.config_utils.compose')
@patch('rembler.utils.config_utils.OmegaConf')
def test_load_config_with_compose_api_success(mock_omega_conf, mock_compose, mock_initialize):
    """Test successful config loading with Compose API."""
    mock_cfg = MagicMock()
    mock_compose.return_value = mock_cfg
    mock_omega_conf.to_yaml.return_value = "model:\n  type: cnn\n"

    with patch('yaml.safe_load', return_value={"model": {"type": "cnn"}}):
        result = load_config_with_compose_api("config", ["param=value"])

    mock_initialize.assert_called_once_with(
        version_base=None,
        config_path="config",
        job_name="test_app"
    )
    mock_compose.assert_called_once_with(
        config_name="config",
        overrides=["param=value"]
    )
    assert result == {"model": {"type": "cnn"}}


@pytest.mark.unit
@patch('rembler.utils.config_utils.initialize')
@patch('rembler.utils.config_utils.compose')
def test_load_config_with_compose_api_defaults(mock_compose, mock_initialize):
    """Test config loading with default parameters."""
    mock_cfg = MagicMock()
    mock_compose.return_value = mock_cfg

    with patch('rembler.utils.config_utils.OmegaConf') as mock_omega_conf:
        mock_omega_conf.to_yaml.return_value = "default: config"
        with patch('yaml.safe_load', return_value={"default": "config"}):
            result = load_config_with_compose_api()

    mock_initialize.assert_called_once_with(
        version_base=None,
        config_path="config",
        job_name="test_app"
    )
    mock_compose.assert_called_once_with(
        config_name="config",
        overrides=[]
    )
    assert result == {"default": "config"}


@pytest.mark.unit
@patch('rembler.utils.config_utils.initialize')
def test_load_config_with_compose_api_hydra_error(mock_initialize):
    """Test handling of Hydra initialization errors."""
    mock_initialize.side_effect = Exception("Hydra initialization failed")

    with pytest.raises(Exception, match="Hydra initialization failed"):
        load_config_with_compose_api("invalid_path")


@pytest.mark.unit
def test_unpack_config_returns_none_tuple():
    """Test that unpack_config returns tuple of None values."""
    test_config = {"model": {"type": "cnn"}, "data": {"batch_size": 32}}

    result = unpack_config(test_config)

    assert result == (None, None, None, None, None)
    assert len(result) == 5
    assert all(item is None for item in result)


@pytest.mark.unit
def test_unpack_config_with_empty_dict():
    """Test unpack_config with empty dictionary."""
    result = unpack_config({})

    assert result == (None, None, None, None, None)


@pytest.mark.unit
def test_unpack_config_with_none():
    """Test unpack_config with None input."""
    result = unpack_config(None)

    assert result == (None, None, None, None, None)


@pytest.mark.integration
def test_load_config_from_hydra_dump_integration():
    """Integration test with real temporary file."""
    test_config = {
        "model": {"type": "cnn", "layers": 3},
        "training": {"epochs": 100, "lr": 0.001}
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the expected directory structure
        experiment_dir = os.path.join(temp_dir, "test_experiment", ".hydra")
        os.makedirs(experiment_dir)

        config_file = os.path.join(experiment_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)

        result = load_config_from_hydra_dump("test_experiment", temp_dir)

        assert result == test_config


@pytest.mark.integration
def test_config_file_path_construction():
    """Test that file path is constructed correctly."""
    experiment_name = "my_experiment"
    base_dir = "/custom/base/path"

    expected_path = f"{base_dir}/{experiment_name}/.hydra/config.yaml"

    with patch("builtins.open", side_effect=FileNotFoundError()) as mock_file:
        with pytest.raises(FileNotFoundError):
            load_config_from_hydra_dump(experiment_name, base_dir)

        mock_file.assert_called_once_with(expected_path, 'r')


@pytest.mark.unit
def test_base_dir_constant():
    """Test that BASE_DIR constant is properly defined."""
    assert BASE_DIR == "/path/to/your/base/dir"
    assert isinstance(BASE_DIR, str)