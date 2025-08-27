"""Set up file for bqskit-ft."""
from __future__ import annotations

from setuptools import find_namespace_packages
from setuptools import setup

# Package configuration now in pyproject.toml
setup(
    packages=find_namespace_packages(exclude=['tests']),
)
