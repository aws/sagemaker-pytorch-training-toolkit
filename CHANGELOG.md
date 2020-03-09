# Changelog

## v1.2.1 (2020-03-09)

### Bug Fixes and Other Changes

 * Install awscli from pypi instead of conda for PyTorch containers

## v1.2.0 (2020-02-27)

### Features

 * Remove unnecessary dependencies.

### Bug Fixes and Other Changes

 * Fix python 2 tox dependencies.
 * copy all tests to test-toolkit folder.
 * Update license URL
 * Adding changes for PyTorch 1.4.0 DLC

## v1.1.0 (2020-02-09)

### Features

 * Add release to PyPI. Change package name to sagemaker-pytorch-training.

### Bug Fixes and Other Changes

 * Fix flake8 erros. Add flake configuration to run during PR.
 * Add twine section to tox.
 * Update build artifacts
 * update: Bump awscli version and constrain spyder on conda
 * update: bump smdebug version to 0.5.0.post0
 * Create __init__.py
 * run local GPU tests for Python 3
 * update: Update buildspec for PyTorch 1.3.1
 * update copyright year in license header
 * Added changes for DLC 2.1 with PyTorch v1.3.1
 * Remove stale-bot config
 * upgrade sagemaker-containers to 2.5.11
 * upgrade pillow to 6.2.0
 * use SageMaker Containers' ProcessRunner for executing the entry point
 * use regional endpoint for STS in builds
 * update instance type region availability.
 * Update Dockerfile.gpu
 * Removing extra packages to optimize space
 * Adding function to skip test for py2 verison
 * Installing tochvision from official pip wheel
 * Add /bin/bash as default CMD
 * Pytorch 1.2 py2 py3 dockerfiles added
 * Add wait on entrypoint
 * Add entrypoint script
 * split training and serving logic

### Testing and Release Infrastructure

 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v1.0.9 (2019-08-15)

### Bug fixes and other changes

 * fix placeholder name cpu-instance-type in buildspec-release.yml

## v1.0.8 (2019-08-15)

### Bug fixes and other changes

 * Update no-p2 and no-p3 regions.

## v1.0.7 (2019-08-06)

### Bug fixes and other changes

 * upgrade sagemaker-container version

## v1.0.6 (2019-06-21)

### Bug fixes and other changes

 * unmark 2 deploy tests

## v1.0.5 (2019-06-20)

### Bug fixes and other changes

 * update p2 restricted regions

## v1.0.4 (2019-06-19)

### Bug fixes and other changes

 * skip tests in gpu instance restricted regions

## v1.0.3 (2019-06-18)

### Bug fixes and other changes

 * modify buildspecs and tox files

## v1.0.2 (2019-06-17)

### Bug fixes and other changes

 * freeze dependency versions

## v1.0.1 (2019-06-13)

### Bug fixes and other changes

 * add buildspec-release file and upgrade cuda version
 * upgrade PyTorch to 1.1
 * disable test_mnist_gpu for py2 for now
 * fix broken line of buildspec
 * prevent hidden errors in buildspec
 * Add AWS CodeBuild buildspec for pull request
 * Bump minimum SageMaker Containers version to 2.4.6 and pin SageMaker Python SDK to 1.18.16
 * fix broken link in README
 * Add timeout to test_mnist_gpu test
 * Use dummy role in tests and update local failure integ test
 * Use the SageMaker Python SDK for local serving integ tests
 * Use the SageMaker Python SDK for local integ distributed training tests
 * Use the SageMaker Python SDK for local integ single-machine training tests
 * Pin fastai version to 1.0.39 in CPU dockerfile
 * Use the SageMaker Python SDK for SageMaker integration tests
 * Add missing rendering dependencies for opencv and a simple test.
 * Add opencv support.
 * Freeze PyYAML version to avoid conflict with Docker Compose
 * Unfreeze numpy version.
 * Freeze TorchVision to 0.2.1
 * Specify region when creating S3 resource in integ tests
 * Read framework version from Python SDK for integ test default
 * Fix unicode display problem in py2 container
 * freeze pip <=18.1, fastai == 1.0.39, numpy <= 1.15.4
 * Add support for fastai (https://github.com/fastai/fastai) library.
 * Remove "requsests" from tests dependencies to avoid regular conflicts with "requests" package from "sagemaker" dependencies.
 * Add support for PyTorch-1.0.
