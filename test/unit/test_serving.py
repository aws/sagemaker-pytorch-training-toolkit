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

import csv
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from mock import MagicMock
from mock import patch
from sagemaker_containers.beta.framework import content_types, errors
from six import StringIO, BytesIO
from torch.autograd import Variable

from sagemaker_pytorch_container.serving import main, default_model_fn, default_input_fn
from sagemaker_pytorch_container.serving import default_predict_fn, default_output_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyModel(nn.Module):
    def __init__(self, ):
        super(DummyModel, self).__init__()

    def forward(self, x):
        pass

    def __call__(self, tensor):
        return 3 * tensor


@pytest.fixture(scope='session', name='tensor')
def fixture_tensor():
    tensor = torch.rand(5, 10, 7, 9)
    return tensor.to(device)


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        default_model_fn('model_dir')


def test_default_input_fn_json(tensor):
    json_data = json.dumps(tensor.cpu().numpy().tolist())
    deserialized_np_array = default_input_fn(json_data, content_types.JSON)

    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert torch.equal(tensor, deserialized_np_array)


def test_default_input_fn_csv():
    array = [[1, 2, 3], [4, 5, 6]]
    str_io = StringIO()
    csv.writer(str_io, delimiter=',').writerows(array)

    deserialized_np_array = default_input_fn(str_io.getvalue(), content_types.CSV)

    tensor = torch.FloatTensor(array).to(device)
    assert torch.equal(tensor, deserialized_np_array)
    assert deserialized_np_array.is_cuda == torch.cuda.is_available()


def test_default_input_fn_csv_bad_columns():
    str_io = StringIO()
    csv_writer = csv.writer(str_io, delimiter=',')
    csv_writer.writerow([1, 2, 3])
    csv_writer.writerow([1, 2, 3, 4])

    with pytest.raises(ValueError):
        default_input_fn(str_io.getvalue(), content_types.CSV)


def test_default_input_fn_npy(tensor):
    stream = BytesIO()
    np.save(stream, tensor.cpu().numpy())
    deserialized_np_array = default_input_fn(stream.getvalue(), content_types.NPY)

    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert torch.equal(tensor, deserialized_np_array)


def test_default_input_fn_bad_content_type():
    with pytest.raises(errors.UnsupportedFormatError):
        default_input_fn('', 'application/not_supported')


def test_default_predict_fn(tensor):
    model = DummyModel()
    prediction = default_predict_fn(tensor, model)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda == torch.cuda.is_available()


def test_default_predict_fn_cpu_cpu(tensor):
    prediction = default_predict_fn(tensor.cpu(), DummyModel().cpu())

    model = DummyModel().to(device)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda == torch.cuda.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_predict_fn_cpu_gpu(tensor):
    model = DummyModel().cuda()
    prediction = default_predict_fn(tensor.cpu(), model)
    assert torch.equal(model(tensor), prediction)
    assert prediction.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_predict_fn_gpu_cpu(tensor):
    prediction = default_predict_fn(tensor.cpu(), DummyModel().cpu())
    model = DummyModel().cuda()
    assert torch.equal(model(tensor), prediction)
    assert prediction.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_predict_fn_gpu_gpu(tensor):
    tensor = tensor.cuda()
    model = DummyModel().cuda()
    prediction = default_predict_fn(tensor, model)
    assert torch.equal(model(tensor), prediction)
    assert prediction.is_cuda is True


def test_default_output_fn_json(tensor):
    output = default_output_fn(tensor, content_types.JSON)

    assert json.dumps(tensor.cpu().numpy().tolist()) in output.get_data(as_text=True)
    assert content_types.JSON == output.mimetype


def test_default_output_fn_npy(tensor):
    output = default_output_fn(tensor, content_types.NPY)

    stream = BytesIO()
    np.save(stream, tensor.cpu().numpy())

    assert stream.getvalue() in output.get_data(as_text=False)
    assert content_types.NPY == output.mimetype


def test_default_output_fn_csv_long():
    tensor = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    output = default_output_fn(tensor, content_types.CSV)

    assert '1,2,3\n4,5,6\n' in output.get_data(as_text=True)
    assert content_types.CSV == output.mimetype


def test_default_output_fn_csv_float():
    tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    output = default_output_fn(tensor, content_types.CSV)

    assert '1.0,2.0,3.0\n4.0,5.0,6.0\n' in output.get_data(as_text=True)
    assert content_types.CSV == output.mimetype


def test_default_output_fn_bad_accept():
    with pytest.raises(errors.UnsupportedFormatError):
        default_output_fn('', 'application/not_supported')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_output_fn_gpu():
    tensor_gpu = torch.LongTensor([[1, 2, 3], [4, 5, 6]]).cuda()

    output = default_output_fn(tensor_gpu, content_types.CSV)

    assert '1,2,3\n4,5,6\n' in output.get_data(as_text=True)
    assert content_types.CSV == output.mimetype


@patch('sagemaker_containers.beta.framework.modules.import_module')
@patch('sagemaker_containers.beta.framework.worker.Worker')
@patch('sagemaker_containers.beta.framework.transformer.Transformer.initialize')
@patch('sagemaker_containers.beta.framework.env.ServingEnv', MagicMock())
def test_hosting_start(mock_import_module, mock_worker, mock_transformer_init):
    environ = MagicMock()
    start_response = MagicMock()
    main(environ, start_response)
    mock_transformer_init.assert_called()
    mock_worker.return_value.assert_called_with(environ, start_response)
