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
from test.integration import dist_operations_path, fastai_path
from test.integration.sagemaker.estimator import PytorchTestEstimator
from test.integration.sagemaker.timeout import timeout


@pytest.mark.skip_gpu
def test_dist_operations_cpu(sagemaker_session, ecr_image, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_cpu_backend)


@pytest.mark.skip_cpu
def test_dist_operations_gpu(sagemaker_session, instance_type, ecr_image, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_gpu_backend)


@pytest.mark.skip_cpu
def test_dist_operations_multi_gpu(sagemaker_session, ecr_image, dist_gpu_backend):
    instance_type = 'ml.p3.8xlarge'
    _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_gpu_backend, 1)


@pytest.mark.skip_cpu
def test_dist_operations_fastai_gpu(sagemaker_session, ecr_image, py_version):
    if py_version != 'py3':
        print('Skipping the test because fastai supports >= Python 3.6.')
        return

    instance_type = 'ml.p3.8xlarge'
    with timeout(minutes=8):
        pytorch = PytorchTestEstimator(entry_point='train_cifar.py',
                                       source_dir=os.path.join(fastai_path, 'cifar'),
                                       role='SageMakerRole',
                                       train_instance_count=1,
                                       train_instance_type=instance_type,
                                       sagemaker_session=sagemaker_session,
                                       docker_image_uri=ecr_image)
        pytorch.sagemaker_session.default_bucket()
        training_nput = pytorch.sagemaker_session.upload_data(
            path=os.path.join(fastai_path, 'cifar_tiny', 'training'),
            key_prefix='pytorch/distributed_operations'
        )
        pytorch.fit({'training': training_nput})


def _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_backend, train_instance_count=3):
    with timeout(minutes=8):
        pytorch = PytorchTestEstimator(entry_point=dist_operations_path, role='SageMakerRole',
                                       train_instance_count=train_instance_count, train_instance_type=instance_type,
                                       sagemaker_session=sagemaker_session, docker_image_uri=ecr_image,
                                       hyperparameters={'backend': dist_backend})
        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(path=dist_operations_path,
                                                           key_prefix='pytorch/distributed_operations')
        pytorch.fit({'required_argument': fake_input})
