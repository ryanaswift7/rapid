# conftest.py
import pytest
import torch

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: integration tests that require external resources"
    )
    config.addinivalue_line(
        "markers", "slow: tests that are slow to run"
    )

@pytest.fixture(scope="session")
def device():
    """Provides a device for tests to ensure consistency."""
    return torch.device("cpu")