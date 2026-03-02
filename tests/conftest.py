import pytest

@pytest.fixture(scope="session")
def session_scoped_fixture():
    # Session-scoped Fixture für Tests
    pass

@pytest.mark.slow
def test_example_slow():
    # Beispiel für einen langsamen Test
    pass

@pytest.mark.integration
def test_example_integration():
    # Beispiel für einen Integrationstest
    pass

@pytest.mark.performance
def test_example_performance():
    # Beispiel für einen Performance-Test
    pass
