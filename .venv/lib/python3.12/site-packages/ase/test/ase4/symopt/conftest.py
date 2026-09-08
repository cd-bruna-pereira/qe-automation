import pytest


@pytest.fixture(autouse=True)
def requires_spglib():
    pytest.importorskip('spglib')
