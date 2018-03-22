import os
import pytest
import torch
import utils
from os.path import join
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dir_path = join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources', 'mnist')
data_dir = join(dir_path, 'data')
training_dir = os.path.join(data_dir, 'training')

mnist_script = join(dir_path, 'mnist.py')

ENTRYPOINT = ["python", "-m", "pytorch_container.start"]


def test_mnist_cpu(image_name, opt_ml):
    utils.train(mnist_script, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT)

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_mnist_gpu(image_name, opt_ml):
    utils.train(mnist_script, data_dir, image_name(device='gpu'), opt_ml, entrypoint=ENTRYPOINT)

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'
