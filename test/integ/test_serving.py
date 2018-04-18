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
import json
from six import StringIO, BytesIO
import pytest
import requests
from test.utils import local_mode
import torch
import logging
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import numpy as np
from container_support.serving import JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, NPY_CONTENT_TYPE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

mnist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources', 'mnist')
mnist_script = os.path.join(mnist_path, 'mnist.py')
model_cpu_dir = os.path.join(mnist_path, 'model_cpu')
model_gpu_dir = os.path.join(mnist_path, 'model_gpu')

data_dir = os.path.join(mnist_path, 'data')
training_dir = os.path.join(data_dir, 'training')

ENTRYPOINT = ["python", "-m", "pytorch_container.start"]


@pytest.fixture(name='serving_cpu')
def fixture_serving_cpu(docker_image, opt_ml):
    return local_mode.serve(customer_script=mnist_script, model_dir=model_cpu_dir, image_name=docker_image,
                            opt_ml=opt_ml, entrypoint=ENTRYPOINT)


@pytest.fixture(name='serving_gpu')
def fixture_serving_gpu(docker_image, opt_ml):
    return local_mode.serve(customer_script=mnist_script, model_dir=model_cpu_dir, image_name=docker_image,
                            use_gpu=True, opt_ml=opt_ml, entrypoint=ENTRYPOINT)


def test_small_batch_data_cpu(serving_cpu):
    with serving_cpu:
        _assert_prediction(batch_size=2)


def test_large_batch_data_cpu(serving_cpu):
    with serving_cpu:
        _assert_prediction(batch_size=300)  # client_max_body_size 5m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_small_batch_data_gpu(serving_gpu):
    with serving_gpu:
        _assert_prediction(batch_size=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_large_batch_data_gpu(serving_gpu):
    with serving_gpu:
        _assert_prediction(batch_size=300)  # client_max_body_size 5m


def _get_test_data_loader(batch_size):
    logger.info('training dir: {}'.format(os.listdir(training_dir)))
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)


def _test_model(docker_image, opt_ml):
    local_mode.train(mnist_script, data_dir, docker_image, opt_ml, entrypoint=ENTRYPOINT)


def _assert_prediction(batch_size):
    test_loader = _get_test_data_loader(batch_size)

   # output = _mnist_prediction(test_loader, JSON_CONTENT_TYPE)
   # assert np.asarray(output).shape == (batch_size, 10)

   # output = _mnist_prediction(test_loader, NPY_CONTENT_TYPE)
   # assert np.asarray(output).shape == (batch_size, 10)

    output = _make_prediction(torch.rand(batch_size, 784), CSV_CONTENT_TYPE, CSV_CONTENT_TYPE)
    assert np.asarray(output).shape == (batch_size, 10)


def _mnist_prediction(test_loader, content_type):
    for data in test_loader:
        return _make_prediction(data[0].numpy(), content_type, content_type)


def _make_prediction(data, request_type, accept):
    serialized_output = requests.post(local_mode.REQUEST_URL, data=_serialize_input(data, request_type),
                                      headers={'Content-type': request_type, 'Accept': accept}).content
    return _deserialize_output(serialized_output, accept)


def _serialize_input(data_to_serialize, content_type):
    if content_type == JSON_CONTENT_TYPE:
        return json.dumps(data_to_serialize.tolist())

    if content_type == CSV_CONTENT_TYPE:
        stream = StringIO()
        np.savetxt(stream, data_to_serialize, delimiter=',', fmt='%s')
        return stream.getvalue()

    if content_type == NPY_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, data_to_serialize)
        return stream.getvalue()


def _deserialize_output(serialized_data, content_type):
    if content_type == JSON_CONTENT_TYPE:
        return np.array(json.loads(serialized_data.decode()))

    if content_type == CSV_CONTENT_TYPE:
        return np.genfromtxt(StringIO(serialized_data), delimiter=',')

    if content_type == NPY_CONTENT_TYPE:
        return np.load(BytesIO(serialized_data))
