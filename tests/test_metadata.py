from pathlib import Path

import pytest

try:
    import tomllib

except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

trove_classifiers = pytest.importorskip("trove_classifiers")


def test_project_classifiers_are_valid_trove_classifiers():
    """Reject classifiers that PyPI would reject during upload."""

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    classifiers = pyproject["project"]["classifiers"]
    invalid = sorted(set(classifiers) - set(trove_classifiers.classifiers))

    assert invalid == []
