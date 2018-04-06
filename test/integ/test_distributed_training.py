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
import os
import pytest
import torch
import utils

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources')
data_dir = os.path.join(dir_path, 'mnist', 'data')
training_dir = os.path.join(data_dir, 'training')

mnist_script = os.path.join(dir_path, 'mnist', 'mnist.py')
dist_operations = os.path.join(dir_path, 'distributed_operations.py')

ENTRYPOINT = ["python", "-m", "pytorch_container.start"]


@pytest.fixture(scope='session', name='dist_cpu_backend', params=['tcp', 'gloo'])
def fixture_dist_cpu_backend(request):
    return request.param


@pytest.fixture(scope='session', name='dist_gpu_backend', params=['gloo'])
def fixture_dist_gpu_backend(request):
    return request.param


def test_dist_operations_cpu(region, image_name, opt_ml, dist_cpu_backend):
    utils.train(region, dist_operations, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT, cluster_size=3,
                hyperparameters={'backend': dist_cpu_backend})

    assert utils.file_exists(opt_ml, 'model/success'), 'Script success file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_dist_operations_gpu(region, image_name, opt_ml, dist_gpu_backend):
    utils.train(region, dist_operations, data_dir, image_name(device='gpu'), opt_ml, entrypoint=ENTRYPOINT, cluster_size=3,
                use_gpu=True, hyperparameters={'backend': dist_gpu_backend})

    assert utils.file_exists(opt_ml, 'model/success'), 'Script success file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_mnist_cpu(region, image_name, opt_ml, dist_cpu_backend):
    utils.train(region, mnist_script, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT, cluster_size=2,
                hyperparameters={'backend': dist_cpu_backend})

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_mnist_gpu(region, image_name, opt_ml, dist_gpu_backend):
    utils.train(region, mnist_script, data_dir, image_name(device='gpu'), opt_ml, entrypoint=ENTRYPOINT, cluster_size=2,
                use_gpu=True, hyperparameters={'backend': dist_gpu_backend})

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'
