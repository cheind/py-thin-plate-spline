# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='thinplate',
    version=open('thinplate/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Thin plate splines for numpy and pytorch',    
    author='Christoph Heindl',
    url='https://github.com/cheind/py-thin-plate-spline',
    license='MIT',
    install_requires=required,
    packages=['thinplate', 'thinplate.tests'],
    include_package_data=True,
    keywords='thin plate spline spatial transformer network'
)