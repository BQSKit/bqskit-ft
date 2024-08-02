"""Set up file for bqskit-shuttling."""
from setuptools import setup, find_namespace_packages

setup(
    name='bqskit-ft',
    description='BQSKit extension for compiling to Fault-Tolerant gate sets.',
    version='0.1.0',
    packages=find_namespace_packages(),
    install_requires=['bqskit', 'numpy', 'scipy'],
    python_requires='>=3.8, <4.0'
)
