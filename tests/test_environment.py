import sys
from pathlib import Path

import pytest

DATASET_DIRECTORY = Path("data/external/5G-QoERA/5G-QoERA")
DATASET_AVAILABLE = DATASET_DIRECTORY.is_dir()


def test_python_version() -> None:
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=("optional 5G-QoERA raw dataset is not installed; " "see README section 7"),
)
def test_dataset_directory_exists() -> None:
    assert DATASET_DIRECTORY.exists()


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=("optional 5G-QoERA raw dataset is not installed; " "see README section 7"),
)
def test_dataset_contains_32_tsv_files() -> None:
    files = list(DATASET_DIRECTORY.glob("*.tsv"))

    assert len(files) == 32
