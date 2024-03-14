import os
from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_ROOT, 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name='biobank_project',
    packages=find_packages(),
    version=version,
    description='Project code for the California State Biobank newborn metabolite screening data',
    author='Alan Chang'
)