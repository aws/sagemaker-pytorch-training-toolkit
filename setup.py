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

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


test_dependencies = ['boto3', 'coverage', 'flake8', 'future', 'mock', 'pytest', 'pytest-cov',
                     'pytest-xdist', 'sagemaker[local]<2', 'torch', 'torchvision', 'tox']

setup(
    name='sagemaker_pytorch_training',
    version=read('VERSION').strip(),
    description='Open source library for using PyTorch to train models on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-pytorch-training-toolkit',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    install_requires=['retrying', 'sagemaker-training>=4.2.0', 'six>=1.12.0'],
    extras_require={
        'test': test_dependencies
    },
)
