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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources', 'mnist')
data_dir = os.path.join(dir_path, 'data')
training_dir = os.path.join(data_dir, 'training')

mnist_script = os.path.join(dir_path, 'mnist.py')

ENTRYPOINT = ["python", "-m", "pytorch_container.start"]


def test_mnist_cpu(image_name, opt_ml):
    utils.train(mnist_script, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT)

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_mnist_gpu(image_name, opt_ml):
    utils.train(mnist_script, data_dir, image_name(device='gpu'), opt_ml, use_gpu=True, entrypoint=ENTRYPOINT)

    assert utils.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert utils.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not utils.file_exists(opt_ml, 'output/failure'), 'Failure happened'
