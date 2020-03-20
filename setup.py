# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from glob import glob
import os
from os.path import basename
from os.path import splitext
import sys

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


test_dependencies = ['boto3==1.10.32', 'coverage==4.5.3', 'docker-compose==1.23.2', 'flake8==3.7.7', 'Flask==1.1.1',
                     'mock==2.0.0', 'pytest==4.4.0', 'pytest-cov==2.7.1', 'pytest-xdist==1.28.0', 'PyYAML==3.10',
                     'sagemaker==1.28.1', 'torch==1.4.0', 'torchvision==0.5.0', 'tox==3.7.0', 'requests_mock==1.6.0']

if sys.version_info.major > 2:
    test_dependencies.append('sagemaker-experiments==0.1.7')

setup(
    name='sagemaker_pytorch_training',
    version=read('VERSION').strip(),
    description='Open source library for creating PyTorch containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-pytorch-container',
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

    install_requires=[
        'retrying',
        'sagemaker-containers>=2.8.2',
        'six>=1.12.0',
        'sagemaker-experiments==0.1.7;python_version>="3.6"'],
    extras_require={
        'test': test_dependencies
    },
)
