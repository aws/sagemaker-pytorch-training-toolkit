from __future__ import absolute_import

import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages
import subprocess


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_pytorch_container',
    version='1.1',
    description='Open source library for creating PyTorch containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    # freeze numpy version because of the python2 bug
    # in 16.0: https://github.com/numpy/numpy/pull/12754
    install_requires=['numpy<=1.15.4', 'Pillow', 'retrying', 'sagemaker-containers>=2.3.5', 'six',
                      'torch==1.0.0'],
    extras_require={
        'test': ['boto3>=1.4.8', 'coverage', 'docker-compose', 'flake8', 'Flask', 'mock',
                 'pytest', 'pytest-cov', 'pytest-xdist', 'PyYAML', 'sagemaker', 'torchvision==0.2.1',
                 'tox']
    },
)
