"""Minimal import test: the atop package and every one of its subpackages
and modules can be imported.

A module that needs an optional third-party dependency which is not
installed in the current environment (for example torch on a CPU-only
build machine) is reported as skipped. Any other import failure, including
a missing atop module, fails the test.

Run with ``python -m pytest tests`` or ``python tests/test_import.py``.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import atop  # noqa: E402

SUBPACKAGES = [
    "atop.attribution",
    "atop.baselines",
    "atop.data",
    "atop.explain",
    "atop.mining",
    "atop.models",
]


def _module_names() -> list[str]:
    names = []
    for info in pkgutil.walk_packages(atop.__path__, prefix="atop."):
        names.append(info.name)
    return sorted(names)


def _try_import(name: str) -> str | None:
    """Import ``name``. Return None on success, or the missing third-party
    module name if an optional dependency is absent. Raise on anything else."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "atop" or missing.startswith("atop."):
            raise
        return missing
    return None


def test_top_level_package():
    assert atop.__version__
    assert set(atop.__all__) == {"AToPExplainer", "AToPConfig"}


def test_subpackages_are_discovered():
    found = set(_module_names())
    for sub in SUBPACKAGES:
        assert sub in found, f"{sub} not found under atop/"


def test_import_subpackages_and_modules():
    skipped = {}
    for name in SUBPACKAGES + _module_names():
        missing = _try_import(name)
        if missing:
            skipped[name] = missing
    if skipped:
        print("skipped (optional dependency not installed):")
        for name, missing in sorted(skipped.items()):
            print(f"  {name}: needs {missing}")


if __name__ == "__main__":
    test_top_level_package()
    test_subpackages_are_discovered()
    test_import_subpackages_and_modules()
    print("test_import: OK")
