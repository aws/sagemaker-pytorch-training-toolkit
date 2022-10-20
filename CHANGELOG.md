# Changelog

## v2.7.0 (2022-10-20)

### Features

 * Add torch_distributed support for Trainium

## v2.6.2.post0 (2022-09-21)

### Documentation Changes

 * update README and contributing guidelines

## v2.6.2 (2022-08-26)

### Bug Fixes and Other Changes

 * provide option to use native process launcher

## v2.6.1 (2022-08-15)

## v2.6.0 (2022-08-03)

### Features

 * add support for native PyTorch DDP distribution

## v2.5.1 (2022-07-21)

### Bug Fixes and Other Changes

 * deriver master node from training environment

## v2.5.0 (2022-07-08)

### Features

 * Add Heterogeneous Cluster support

## v2.4.1 (2022-07-08)

### Bug Fixes and Other Changes

 * CI

## v2.4.0 (2020-12-11)

### Features

 * add data parallelism support (#11) (#12)

### Bug Fixes and Other Changes

 * use ubuntu 18.04 base image in dlc gpu image
 * remove TODOs in 1.6.0 dlc gpu dockerfile and reduce parameters for data parallel integ test
 * use base cuda 11 image for test dlc gpu image
 * use 1.6.0 for gpu tests and disable horovod tests
 * remove local data parallel integ test
 * use sagemaker-training 3.7.0 and enable data parallel integ tests
 * patch socket call and update flake8 violations

## v2.3.0 (2020-08-31)

### Features

 * Use MPIRunnerType

### Bug Fixes and Other Changes

 * Update main buildspec to only perform CPU integration tests
 * Add GPU and unit test buildspecs
 * Pin SageMaker version to less than v2

### Documentation Changes

 * improve training.py doc style

## v2.2.1.post2 (2020-06-25)

### Testing and Release Infrastructure

 * add issue templates

## v2.2.1.post1 (2020-06-16)

### Documentation Changes

 * remove confusing information from the Readme.

### Testing and Release Infrastructure

 * do not duplicate test dependencies in tox.ini
 * Rename buildspec files.

## v2.2.1.post0 (2020-06-05)

### Testing and Release Infrastructure

 * Make docker folder read only, remove unused tests, rename test-toolkit/ -> test/.

## v2.2.1 (2020-05-12)

### Bug Fixes and Other Changes

 * Bump version of sagemaker-training for typing fix

## v2.2.0 (2020-05-07)

### Features

 * add Python 3.7 support

## v2.1.1 (2020-05-05)

### Bug Fixes and Other Changes

 * Pin Smdebug to the latest version (0.7.2)

## v2.1.0 (2020-05-04)

### Features

 * add Dockerfiles for PyTorch 1.5.0

## v2.0.0 (2020-04-27)

### Breaking Changes

 * Replace sagemaker-containers with sagemaker-training

## v1.3.3 (2020-04-16)

### Bug Fixes and Other Changes

 * change miniconda installation in 1.4.0 Dockerfiles

### Testing and Release Infrastructure

 * parallelize SageMaker integ test runs
 * remove (unused) model_fn from training scripts

## v1.3.2 (2020-04-07)

### Bug Fixes and Other Changes

 * bump smdebug version

### Testing and Release Infrastructure

 * add requirements.txt integ test

## v1.3.1 (2020-04-02)

### Bug Fixes and Other Changes

 * upgrade pillow etc. to fix safety issues
 * Upgrade sagemaker-containers and test with more than 1 epoch

## v1.3.0 (2020-03-23)

### Features

 * Install toolkit from PyPI.

### Bug Fixes and Other Changes

 * upgrade sagemaker-containers to 2.8.2
 * Install jupyter_client 5.3.4 in advanced for py2 gpu image
 * update smdebug

### Testing and Release Infrastructure

 * run test-toolkit unit tests for release
 * run build steps only when necessary.
 * refactor toolkit tests.

## v1.2.4 (2020-03-12)

### Bug Fixes and Other Changes

 * install sm experiments always when python 3.6 or greater

## v1.2.3 (2020-03-11)

### Bug Fixes and Other Changes

 * Update smdebug to 0.7.0
 * install sagemaker-experiments package only for 3.6

## v1.2.2 (2020-03-10)

### Bug Fixes and Other Changes

 * upgrade to latest sagemaker-experiments
 * install SageMaker Python SDK into Python 3 images

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
