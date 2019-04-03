# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import pytest
from sagemaker.pytorch import PyTorch

from test.integration import data_dir, dist_operations_path, mnist_script
from test.utils.local_mode_utils import assert_files_exist

MODEL_SUCCESS_FILES = {
    'model': ['success'],
    'output': ['success'],
}


@pytest.fixture(scope='session', name='dist_gpu_backend', params=['gloo'])
def fixture_dist_gpu_backend(request):
    return request.param


@pytest.mark.skip_gpu
def test_dist_operations_path_cpu(docker_image, dist_cpu_backend, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=dist_operations_path,
                        role='SageMakerRole',
                        image_name=docker_image,
                        train_instance_count=2,
                        train_instance_type='local',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'backend': dist_cpu_backend},
                        output_path='file://{}'.format(tmpdir))

    _train_and_assert_success(estimator, str(tmpdir))


@pytest.mark.skip_cpu
def test_dist_operations_path_gpu(docker_image, dist_gpu_backend, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=dist_operations_path,
                        role='SageMakerRole',
                        image_name=docker_image,
                        train_instance_count=3,
                        train_instance_type='local_gpu',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'backend': dist_gpu_backend},
                        output_path='file://{}'.format(tmpdir))

    _train_and_assert_success(estimator, str(tmpdir))


@pytest.mark.skip_cpu
def test_dist_operations_path_gpu_nccl(docker_image, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=dist_operations_path,
                        role='SageMakerRole',
                        image_name=docker_image,
                        train_instance_count=1,
                        train_instance_type='local_gpu',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'backend': 'nccl'},
                        output_path='file://{}'.format(tmpdir))

    _train_and_assert_success(estimator, str(tmpdir))


@pytest.mark.skip_gpu
def test_cpu_nccl(docker_image, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=mnist_script,
                        role='SageMakerRole',
                        image_name=docker_image,
                        train_instance_count=2,
                        train_instance_type='local',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'backend': 'nccl'},
                        output_path='file://{}'.format(tmpdir))

    # Local Mode doesn't export model/output artifacts upon failure
    # https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/local/image.py#L133-L141
    with pytest.raises(RuntimeError) as e:
        estimator.fit({'training': 'file://{}'.format(os.path.join(data_dir, 'training'))})

    assert 'Failed to run:' in str(e)
    assert 'Process exited with code: 1' in str(e)


@pytest.mark.skip_gpu
def test_mnist_cpu(docker_image, dist_cpu_backend, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=mnist_script,
                        role='SageMakerRole',
                        image_name=docker_image,
                        train_instance_count=2,
                        train_instance_type='local',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'backend': dist_cpu_backend},
                        output_path='file://{}'.format(tmpdir))

    success_files = {
        'model': ['model.pth'],
        'output': ['success'],
    }
    _train_and_assert_success(estimator, str(tmpdir), success_files)


@pytest.mark.skip_cpu
def test_mnist_gpu(docker_image, dist_gpu_backend, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=mnist_script,
                        role='SageMakerRole',
                        image_name=docker_image,
                        train_instance_count=2,
                        train_instance_type='local_gpu',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'backend': dist_gpu_backend},
                        output_path='file://{}'.format(tmpdir))

    success_files = {
        'model': ['model.pth'],
        'output': ['success'],
    }
    _train_and_assert_success(estimator, str(tmpdir), success_files)


def _train_and_assert_success(estimator, output_path, output_files=MODEL_SUCCESS_FILES):
    estimator.fit({'training': 'file://{}'.format(os.path.join(data_dir, 'training'))})
    assert_files_exist(output_path, output_files)
