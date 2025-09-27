from unittest.mock import patch, MagicMock
import pytest
import torch

from rembler.utils.hardware_utils import resolve_device


@pytest.mark.unit
@patch('torch.cuda.is_available')
def test_resolve_device_auto_cuda_available(mock_cuda_available):
    """Test resolve_device returns CUDA when available and auto is specified."""
    mock_cuda_available.return_value = True

    result = resolve_device("auto")

    assert result == torch.device("cuda")
    mock_cuda_available.assert_called_once()


@pytest.mark.unit
@patch('torch.cuda.is_available')
@patch('torch.backends.mps.is_available')
def test_resolve_device_auto_mps_available(mock_mps_available, mock_cuda_available):
    """Test resolve_device returns MPS when CUDA unavailable but MPS available."""
    mock_cuda_available.return_value = False
    mock_mps_available.return_value = True

    result = resolve_device("auto")

    assert result == torch.device("mps")
    mock_cuda_available.assert_called_once()
    mock_mps_available.assert_called_once()


@pytest.mark.unit
@patch('torch.cuda.is_available')
@patch('torch.backends.mps.is_available')
def test_resolve_device_auto_cpu_fallback(mock_mps_available, mock_cuda_available):
    """Test resolve_device falls back to CPU when neither CUDA nor MPS available."""
    mock_cuda_available.return_value = False
    mock_mps_available.return_value = False

    result = resolve_device("auto")

    assert result == torch.device("cpu")
    mock_cuda_available.assert_called_once()
    mock_mps_available.assert_called_once()


@pytest.mark.unit
def test_resolve_device_explicit_cuda():
    """Test resolve_device with explicit CUDA specification."""
    result = resolve_device("cuda")

    assert result == torch.device("cuda")


@pytest.mark.unit
def test_resolve_device_explicit_cpu():
    """Test resolve_device with explicit CPU specification."""
    result = resolve_device("cpu")

    assert result == torch.device("cpu")


@pytest.mark.unit
def test_resolve_device_explicit_mps():
    """Test resolve_device with explicit MPS specification."""
    result = resolve_device("mps")

    assert result == torch.device("mps")


@pytest.mark.unit
def test_resolve_device_explicit_cuda_with_index():
    """Test resolve_device with specific CUDA device index."""
    result = resolve_device("cuda:1")

    assert result == torch.device("cuda:1")


@pytest.mark.unit
def test_resolve_device_default_parameter():
    """Test resolve_device with default 'auto' parameter."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        result = resolve_device()

        assert result == torch.device("cpu")


@pytest.mark.unit
def test_resolve_device_invalid_device_string():
    """Test resolve_device with invalid device string."""
    # torch.device should handle invalid strings by raising an error
    with pytest.raises(RuntimeError):
        resolve_device("invalid_device")


@pytest.mark.unit
@patch('torch.cuda.is_available')
def test_resolve_device_cuda_priority_over_mps(mock_cuda_available):
    """Test that CUDA is preferred over MPS when both are available."""
    mock_cuda_available.return_value = True

    with patch('torch.backends.mps.is_available', return_value=True):
        result = resolve_device("auto")

        assert result == torch.device("cuda")
        # MPS availability should not even be checked if CUDA is available
        mock_cuda_available.assert_called_once()


@pytest.mark.unit
@patch('torch.cuda.is_available')
@patch('torch.backends.mps.is_available', side_effect=AttributeError)
def test_resolve_device_mps_not_supported(mock_mps_available, mock_cuda_available):
    """Test graceful handling when MPS is not supported (older PyTorch versions)."""
    mock_cuda_available.return_value = False

    result = resolve_device("auto")

    # Should fall back to CPU if MPS check raises AttributeError
    assert result == torch.device("cpu")


@pytest.mark.unit
def test_resolve_device_return_type():
    """Test that resolve_device always returns a torch.device object."""
    devices_to_test = ["cpu", "cuda", "mps"]

    for device_str in devices_to_test:
        try:
            result = resolve_device(device_str)
            assert isinstance(result, torch.device)
            assert str(result).startswith(device_str)
        except RuntimeError:
            # Some devices might not be available, which is fine for this test
            pass


@pytest.mark.integration
def test_resolve_device_real_hardware():
    """Integration test with actual hardware detection."""
    # Test that auto detection works with real hardware
    result = resolve_device("auto")

    assert isinstance(result, torch.device)
    # Should be one of the supported device types
    assert str(result) in ["cpu"] or str(result).startswith(("cuda", "mps"))


@pytest.mark.integration
def test_resolve_device_consistency():
    """Test that multiple calls to resolve_device with same input return same result."""
    result1 = resolve_device("auto")
    result2 = resolve_device("auto")

    assert result1 == result2
    assert str(result1) == str(result2)