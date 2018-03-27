import os
from os.path import join
import pytest
import torch
import utils

dir_path = join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources')
data_dir = join(dir_path, 'mnist', 'data')
training_dir = os.path.join(data_dir, 'training')

mnist_script = join(dir_path, 'mnist', 'mnist.py')
dist_operations = join(dir_path, 'distributed_operations.py')

ENTRYPOINT = ["python", "-m", "pytorch_container.start"]


@pytest.fixture(scope='session', params=['tcp', 'gloo'])
def _dist_cpu_backend(request):
    return request.param


@pytest.fixture(scope='session', params=['gloo'])
def _dist_gpu_backend(request):
    return request.param


def test_dist_operations_cpu(image_name, opt_ml, _dist_cpu_backend):
    utils.train(dist_operations, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT, cluster_size=3,
                hyperparameters={'backend': _dist_cpu_backend})

    assert utils.file_exists(opt_ml, 'model/success'), 'Script success file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_dist_operations_gpu(image_name, opt_ml, _dist_gpu_backend):
    utils.train(dist_operations, data_dir, image_name(device='gpu'), opt_ml, entrypoint=ENTRYPOINT, cluster_size=3,
                hyperparameters={'backend': _dist_gpu_backend})

    assert utils.file_exists(opt_ml, 'model/success'), 'Script success file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_mnist_cpu(image_name, opt_ml, _dist_cpu_backend):
    utils.train(mnist_script, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT, cluster_size=2,
                hyperparameters={'backend': _dist_cpu_backend})

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_mnist_gpu(image_name, opt_ml, _dist_gpu_backend):
    utils.train(mnist_script, data_dir, image_name(device='gpu'), opt_ml, entrypoint=ENTRYPOINT, cluster_size=2,
                hyperparameters={'backend': _dist_gpu_backend})

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'
