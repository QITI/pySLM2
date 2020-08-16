import pytest
import pySLM2


def pytest_addoption(parser):
    parser.addoption(
        "--backend64", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    backend64=config.getoption("backend64")

    if backend64:
        pySLM2.BACKEND.dtype = pySLM2.BACKEND.TENSOR_64BITS
        print("Use 64 bit backend...")

