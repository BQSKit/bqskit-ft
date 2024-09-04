"""Set up file for bqskit-ft."""
from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

setup(
    name='bqskit-ft',
    description='BQSKit extension for compiling to Fault-Tolerant gate sets.',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['bqskit', 'numpy', 'scipy'],
    python_requires='>=3.8, <4.0',
)
