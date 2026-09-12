import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ig3d.config import DEFAULT  # noqa: E402
from ig3d.volumes import generate  # noqa: E402


@pytest.fixture(scope="session")
def vols():
    return generate(3, seed=123, cfg=DEFAULT.volume)


@pytest.fixture(scope="session")
def cfg():
    return DEFAULT
