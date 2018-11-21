from __future__ import absolute_import
import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages
import subprocess


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# check if system has CUDA enabled GPU
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
processor = 'cu92' if p.communicate()[0].decode('UTF-8') != '' else 'cpu'
subprocess.check_call(
    [
        'pip install torch_nightly -f '
        'https://download.pytorch.org/whl/nightly/{}/torch_nightly.html'.format(processor)
    ],
    shell=True
)
subprocess.check_call(['pip install --no-cache --no-deps torchvision'], shell=True)


setup(
    name='sagemaker_pytorch_container',
    version='1.0',
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
        'Programming Language :: Python :: 3.5',
    ],

    install_requires=['numpy', 'sagemaker-containers', 'Pillow', 'retrying', 'six'],
    extras_require={
        'test': ['tox', 'flake8', 'coverage', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock',
                 'Flask', 'boto3>=1.4.8', 'docker-compose', 'sagemaker', 'PyYAML']
    },
)
