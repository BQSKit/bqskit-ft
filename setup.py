"""Set up file for bqskit-ft."""
from __future__ import annotations

from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name='bqskit-ft',
    description='BQSKit extension for compiling to Fault-Tolerant gate sets.',
    author='Mathias Weiden',
    author_email='mtweiden@berkeley.edu',
    version='0.1.0',
    packages=find_namespace_packages(exclude=['tests']),
    install_requires=['bqskit', 'numpy', 'scipy'],
    python_requires='>=3.8, <4.0',
)
