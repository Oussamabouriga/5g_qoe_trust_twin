import sys
from pathlib import Path


def test_python_version() -> None:
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


def test_dataset_directory_exists() -> None:
    dataset_directory = Path(
        "data/external/5G-QoERA/5G-QoERA"
    )
    assert dataset_directory.exists()


def test_dataset_contains_32_tsv_files() -> None:
    dataset_directory = Path(
        "data/external/5G-QoERA/5G-QoERA"
    )
    files = list(dataset_directory.glob("*.tsv"))

    assert len(files) == 32
