import pytest


@pytest.mark.skip(reason="This test has to be separated from the other test.")
def test_backend():
    import pySLM2
    assert pySLM2.BACKEND._initialized is False
