#!/usr/bin/env bash

py_version=$(python -c 'import sys; print(sys.version_info.major)')
read -p "Building image for python version: ${py_version}. Press enter to proceed."

# Temporary work around to use private sagemaker-container-support
git clone -b mvs-poc git@github.com:aws/sagemaker-container-support.git
cd sagemaker-container-support && python setup.py bdist_wheel
pip install dist/sagemaker_container_support-1.0-py2.py3-none-any.whl
cd ..

docker build -t base-pytorch:0.3.1-cpu-py${py_version} -f docker/0.3.1/base/Dockerfile.cpu --build-arg py_version=${py_version} .

rm -rf sagemaker-container-support/
