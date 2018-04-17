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
import numpy
from six import StringIO, BytesIO
import pytest
import requests
import torch
import utils
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


def test_small_batch_data_cpu(region, image_name, opt_ml):
    with utils.serve(region, customer_script=mnist_script, model_dir=model_cpu_dir, image_name=image_name(),
                     opt_ml=opt_ml, entrypoint=ENTRYPOINT):
        _assert_prediction(batch_size=2)


def test_large_batch_data_cpu(region, image_name, opt_ml):
    with utils.serve(region, customer_script=mnist_script, model_dir=model_cpu_dir, image_name=image_name(),
                     opt_ml=opt_ml, entrypoint=ENTRYPOINT):
        _assert_prediction(batch_size=300)  # client_max_body_size 5m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_small_batch_data_gpu(region, image_name, opt_ml):
    with utils.serve(region, customer_script=mnist_script, model_dir=model_gpu_dir, image_name=image_name(),
                     opt_ml=opt_ml, entrypoint=ENTRYPOINT):
        _assert_prediction(batch_size=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_large_batch_data_gpu(region, image_name, opt_ml):
    with utils.serve(region, customer_script=mnist_script, model_dir=model_gpu_dir, image_name=image_name(),
                     opt_ml=opt_ml, entrypoint=ENTRYPOINT):
        _assert_prediction(batch_size=300)  # client_max_body_size 5m


def _get_test_data_loader(batch_size):
    logger.info('training dir: {}'.format(os.listdir(training_dir)))
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)


def test_model(region, image_name, opt_ml):
    utils.train(region, mnist_script, data_dir, image_name(), opt_ml, entrypoint=ENTRYPOINT)


def test_csv():
    import pandas
    from six import StringIO
    import numpy

    data = numpy.ones((2, 3, 4)) * 2
    logger.info('serialize to csv: {}'.format(data))
    df = pandas.DataFrame(data.tolist())
    csv_data = df.to_csv()
    #logger.info('csv data: {}'.format(csv_data))

    logger.info('deserialize from csv: {}'.format(csv_data))
    df = pandas.read_csv(StringIO(csv_data))
    logger.info('convert dataframe to numpy array')
    ndarray = numpy.array(df.values, dtype=numpy.float32)
    logger.info('ndarray: {}'.format(ndarray))
    # data = torch.FloatTensor(ndarray)
    data = torch.from_numpy(ndarray)
    logger.info('data: {}'.format(data))


def test_npz():
    logger.info('serialize to npz')
    import numpy
    from six import StringIO

    arr_0 = numpy.zeros((1, 1, 3, 4))
    arr_1 = numpy.ones((1, 1, 3, 4))
    arr_2 = numpy.ones((1, 1, 3, 4)) * 2
    stream = StringIO()
    numpy.savez(stream, arr_2=arr_2, arr_0=arr_0, arr_1=arr_1)

    logger.info('deserialize from npz')
    npzfile = numpy.load(StringIO(stream.getvalue()))
    ndarrays = numpy.concatenate([npzfile[f] for f in npzfile.files])
    data = torch.FloatTensor(ndarrays)
    logger.info('data: {}'.format(data))


def _mnist_prediction(test_loader, content_type):
    for data in test_loader:
        logger.info(data[0])
        logger.info(data[0].numpy())
        return make_prediction(data[0].numpy(), content_type)


def _assert_prediction(batch_size):
    test_loader = _get_test_data_loader(batch_size)

    output = _mnist_prediction(test_loader, JSON_CONTENT_TYPE)
    assert np.asarray(output).shape == (batch_size, 10)

    #output = _mnist_prediction(test_loader, CSV_CONTENT_TYPE)
    #assert np.asarray(output).shape == (batch_size, 10)


def make_prediction(data, request_type, accept):
    serialized_output = requests.post(utils.REQUEST_URL, data=_serialize_input(data, request_type),
                                      headers={'Content-type': request_type, 'Accept': accept}).content
    return _deserialize_output(serialized_output, accept)


def _deserialize_output(serialized_input_data, content_type):
    if content_type == JSON_CONTENT_TYPE:
        return np.array(json.loads(serialized_input_data), dtype=np.float32)

    if content_type == CSV_CONTENT_TYPE:
        return np.genfromtxt(StringIO(serialized_input_data), dtype=np.float32, delimiter=',')

    if content_type == NPY_CONTENT_TYPE:
        return np.load(BytesIO(serialized_input_data))


def _serialize_input(prediction_output, content_type):
    if content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output.tolist())

    if content_type == CSV_CONTENT_TYPE:
        stream = StringIO()
        np.savetxt(stream, prediction_output, delimiter=',', fmt='%s')
        return stream.getvalue()

    if content_type == NPY_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue()
