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
import numpy as np
import os
import pytest
from test.integration.sagemaker.test_estimator import PytorchTestEstimator
from test.integration.sagemaker.timeout import timeout, timeout_and_delete_endpoint

mnist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'mnist'))
mnist_script = os.path.join(mnist_path, 'mnist.py')
data_dir = os.path.join(mnist_path, 'data')
training_dir = os.path.join(data_dir, 'training')
mnist_1d_script = os.path.join(mnist_path, 'mnist_1d.py')
model_cpu_dir = os.path.join(mnist_path, 'model_cpu')
model_cpu_1d_dir = os.path.join(model_cpu_dir, '1d')
model_gpu_dir = os.path.join(mnist_path, 'model_gpu')
model_gpu_1d_dir = os.path.join(model_gpu_dir, '1d')


@pytest.mark.skip_gpu
def test_mnist_distributed_cpu(sagemaker_session, ecr_image, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_cpu_backend)


@pytest.mark.skip_cpu
def test_mnist_distributed_gpu(sagemaker_session, ecr_image, instance_type, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_gpu_backend)


def _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_backend):
    with timeout(minutes=10):
        pytorch = PytorchTestEstimator(entry_point=mnist_script, role='SageMakerRole',
                                       train_instance_count=2, train_instance_type=instance_type,
                                       sagemaker_session=sagemaker_session, docker_image_uri=ecr_image,
                                       hyperparameters={'backend': dist_backend, 'epochs': 1})
        fake_input = pytorch.sagemaker_session.upload_data(path=training_dir,
                                                           key_prefix='pytorch/mnist')
        pytorch.fit({'required_argument': fake_input})

    with timeout_and_delete_endpoint(estimator=pytorch, minutes=30):
        predictor = pytorch.deploy(initial_instance_count=1, instance_type=instance_type)

        batch_size = 100
        data = np.zeros(shape=(batch_size, 1, 28, 28))
        output = predictor.predict(data)
    assert len(output) == batch_size
