import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Ensure standard test environment variables are populated."""
    os.environ.setdefault("API_KEY", "test-api-key")
    os.environ.setdefault("SECRET_KEY", "test-secret-key-32-chars-long--")
