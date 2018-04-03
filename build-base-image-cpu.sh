#!/usr/bin/env bash

# Temporary work around to use private sagemaker-container-support
git clone -b mvs-poc git@github.com:aws/sagemaker-container-support.git
cd sagemaker-container-support && python setup.py bdist_wheel
pip install dist/sagemaker_container_support-1.0-py2.py3-none-any.whl
cd ..

default_py_version=2
read -p "Enter python version to use (default is ${default_py_version}):" py_version
py_version=${py_version:-default_py_version}
echo $py_version

docker build -t base-pytorch:0.3.1-cpu-py${py_version} -f docker/0.3.1/base/Dockerfile.cpu --build-arg py_version=${py_version} .

rm -rf sagemaker-container-support/
