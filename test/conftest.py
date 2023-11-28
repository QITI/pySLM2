import pytest
import pySLM2


def pytest_addoption(parser):
    parser.addoption(
        "--backend64", action="store_true", default=False, help="Test with 64bit precision."
    )
    parser.addoption(
        "--luxbeam-ip", action="store", default=None, help="ip address of the DMD Luxbeam Controller",
    )
    parser.addoption(
        "--alp-version", action="store", default=None, help="Version of the ALP library fom Vialux",
    )

@pytest.fixture
def luxbeam_ip(request):
    return request.config.getoption("--luxbeam-ip")

@pytest.fixture
def alp_version(request):
    return request.config.getoption("--alp-version", default="4.3")

def pytest_configure(config):
    backend64 = config.getoption("backend64")

    if backend64:
        pySLM2.BACKEND.dtype = pySLM2.BACKEND.TENSOR_64BITS
        print("Use 64 bit backend...")


    config.addinivalue_line("markers", "luxbeam: mark test that requires luxbeam controller.")
    config.addinivalue_line("markers", "alp: mark test that requires ALP controller.")



def pytest_collection_modifyitems(config, items):
    if config.getoption("--luxbeam-ip") is None:
        skip_luxbeam = pytest.mark.skip(reason="need --luxbeam-ip option to run")
        for item in items:
            if "luxbeam" in item.keywords:
                item.add_marker(skip_luxbeam)

    if config.getoption("--alp", default=False):
        return
    skip_alp = pytest.mark.skip(reason="need --alp option to run")
    for item in items:
        if "alp" in item.keywords:
            item.add_marker(skip_alp)