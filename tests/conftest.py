"""Shared pytest fixtures and helpers for OPENQANT tests."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return repository root path.

    Returns:
        Absolute path to project root.
    """
    return Path(__file__).resolve().parents[1]


def load_module(module_path: Path, module_name: str) -> ModuleType:
    """Load a Python module from a filesystem path.

    Args:
        module_path: Absolute path to `.py` file.
        module_name: Synthetic module name used for loading.

    Returns:
        Loaded module object.

    Raises:
        ImportError: If module cannot be loaded.
    """
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
