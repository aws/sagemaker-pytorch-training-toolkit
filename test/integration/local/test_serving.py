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
import json
from six import StringIO, BytesIO
import pytest
import requests
from test.utils import local_mode
import torch
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import numpy as np
from sagemaker_containers.beta.framework import content_types
from test.integration import training_dir, mnist_script, mnist_1d_script, model_cpu_dir, model_gpu_dir, \
    model_cpu_1d_dir


@pytest.fixture(name='serve_cpu')
def fixture_serve_cpu(docker_image, opt_ml):
    def serve(model_dir=model_cpu_dir, script=mnist_script):
        return local_mode.serve(customer_script=script, model_dir=model_dir, image_name=docker_image,
                                opt_ml=opt_ml)
    return serve


@pytest.fixture(name='serve_gpu')
def fixture_serve_gpu(docker_image, opt_ml):
    def serve(model_dir=model_gpu_dir, script=mnist_script):
        return local_mode.serve(customer_script=script, model_dir=model_dir, image_name=docker_image,
                                use_gpu=True, opt_ml=opt_ml)
    return serve


@pytest.fixture(name='test_loader')
def fixture_test_loader():
    #  Largest batch size is only 300 because client_max_body_size is 5M
    return _get_test_data_loader(batch_size=300)


def test_serve_cpu_json_npy(serve_cpu, test_loader):
    with serve_cpu():
        _assert_prediction_npy_json(test_loader, content_types.JSON, content_types.JSON)
        _assert_prediction_npy_json(test_loader, content_types.JSON, content_types.CSV)
        _assert_prediction_npy_json(test_loader, content_types.JSON, content_types.NPY)

        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.JSON)
        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.CSV)
        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.NPY)


def test_serve_cpu_csv(serve_cpu, test_loader):
    with serve_cpu(model_dir=model_cpu_1d_dir, script=mnist_1d_script):
        _assert_prediction_csv(test_loader, content_types.JSON)
        _assert_prediction_csv(test_loader, content_types.CSV)
        _assert_prediction_csv(test_loader, content_types.NPY)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_serve_gpu_json_npy(serve_gpu, test_loader):
    with serve_gpu():
        _assert_prediction_npy_json(test_loader, content_types.JSON, content_types.JSON)
        _assert_prediction_npy_json(test_loader, content_types.JSON, content_types.CSV)
        _assert_prediction_npy_json(test_loader, content_types.JSON, content_types.NPY)

        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.JSON)
        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.CSV)
        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.NPY)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_serve_gpu_csv(serve_gpu, test_loader):
    with serve_gpu(model_dir=model_cpu_1d_dir, script=mnist_1d_script):
        _assert_prediction_csv(test_loader, content_types.JSON)
        _assert_prediction_csv(test_loader, content_types.CSV)
        _assert_prediction_csv(test_loader, content_types.NPY)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_serve_cpu_model_on_gpu(serve_gpu, test_loader):
    with serve_gpu(model_dir=model_cpu_1d_dir, script=mnist_1d_script):
        _assert_prediction_npy_json(test_loader, content_types.NPY, content_types.JSON)


def _assert_prediction_npy_json(test_loader, request_type, accept):
    output = _make_prediction(_get_mnist_batch(test_loader).numpy(), request_type, accept)
    assert np.asarray(output).shape == (test_loader.batch_size, 10)


def _assert_prediction_csv(test_loader, accept):
    data = _get_mnist_batch(test_loader).view(test_loader.batch_size, -1)
    output = _make_prediction(data, content_types.CSV, accept)
    assert np.asarray(output).shape == (test_loader.batch_size, 10)


def _get_test_data_loader(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)


def _get_mnist_batch(test_loader):
    for data in test_loader:
        return data[0]


def _make_prediction(data, request_type, accept):
    serialized_output = requests.post(local_mode.REQUEST_URL, data=_serialize_input(data, request_type),
                                      headers={'Content-type': request_type, 'Accept': accept}).content
    return _deserialize_output(serialized_output, accept)


def _serialize_input(data_to_serialize, content_type):
    if content_type == content_types.JSON:
        return json.dumps(data_to_serialize.tolist())

    if content_type == content_types.CSV:
        stream = StringIO()
        np.savetxt(stream, data_to_serialize, delimiter=',', fmt='%s')
        return stream.getvalue()

    if content_type == content_types.NPY:
        stream = BytesIO()
        np.save(stream, data_to_serialize)
        return stream.getvalue()


def _deserialize_output(serialized_data, content_type):
    if content_type == content_types.JSON:
        return np.array(json.loads(serialized_data.decode()))

    if content_type == content_types.CSV:
        return np.genfromtxt(StringIO(serialized_data.decode()), delimiter=',')

    if content_type == content_types.NPY:
        return np.load(BytesIO(serialized_data))
