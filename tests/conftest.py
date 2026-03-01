import pytest

from privacy_and_grokking.utils.logger import Logger


@pytest.fixture(autouse=True)
def logger():
    with Logger() as log:
        yield log
