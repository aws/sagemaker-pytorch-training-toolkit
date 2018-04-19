import pytest
import json
import csv
import numpy as np
from six import StringIO, BytesIO
import torch
import torch.nn as nn
from torch.autograd import Variable

from container_support.serving import JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, NPY_CONTENT_TYPE, \
    UnsupportedContentTypeError, UnsupportedAcceptTypeError

from pytorch_container.serving import model_fn, input_fn, predict_fn, output_fn


class DummyModel(nn.Module):
    def __init__(self, ):
        super(DummyModel, self).__init__()

    def forward(self, x):
        pass

    def __call__(self, variable):
        return 3 * variable


@pytest.fixture(scope='session', name='tensor')
def _tensor():
    tensor = torch.rand(5, 10, 7, 9)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def test_model_fn():
    with pytest.raises(NotImplementedError):
        model_fn('model_dir')


def test_input_fn_json(tensor):
    json_data = json.dumps(tensor.cpu().numpy().tolist())
    deserialized_np_array = input_fn(json_data, JSON_CONTENT_TYPE)

    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert torch.equal(tensor, deserialized_np_array)


def test_input_fn_csv():
    array = [[1, 2, 3], [4, 5, 6]]
    str_io = StringIO()
    csv.writer(str_io, delimiter=',').writerows(array)

    deserialized_np_array = input_fn(str_io.getvalue(), CSV_CONTENT_TYPE)

    tensor = torch.cuda.FloatTensor(array) if torch.cuda.is_available() else torch.FloatTensor(array)
    assert torch.equal(tensor, deserialized_np_array)
    assert deserialized_np_array.is_cuda == torch.cuda.is_available()


def test_input_fn_csv_bad_columns():
    str_io = StringIO()
    csv_writer = csv.writer(str_io, delimiter=',')
    csv_writer.writerow([1, 2, 3])
    csv_writer.writerow([1, 2, 3, 4])

    with pytest.raises(ValueError):
        input_fn(str_io.getvalue(), CSV_CONTENT_TYPE)


def test_input_fn_npy(tensor):
    stream = BytesIO()
    np.save(stream, tensor.cpu().numpy())
    deserialized_np_array = input_fn(stream.getvalue(), NPY_CONTENT_TYPE)

    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert torch.equal(tensor, deserialized_np_array)


def test_input_fn_bad_content_type():
    with pytest.raises(UnsupportedContentTypeError):
        input_fn('', 'application/not_supported')


def test_predict_fn(tensor):
    model = DummyModel()
    prediction = predict_fn(tensor, model)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda == torch.cuda.is_available()


def test_predict_fn_cpu_cpu(tensor):
    prediction = predict_fn(tensor.cpu(), DummyModel().cpu())

    model = DummyModel()
    if torch.cuda.is_available():
        model = model.cuda()
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda == torch.cuda.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_predict_fn_cpu_gpu(tensor):
    model = DummyModel().cuda()
    prediction = predict_fn(tensor.cpu(), model)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_predict_fn_gpu_cpu(tensor):
    prediction = predict_fn(tensor.cpu(), DummyModel().cpu())
    model = DummyModel().cuda()
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_predict_fn_gpu_gpu(tensor):
    tensor = tensor.cuda()
    model = DummyModel().cuda()
    prediction = predict_fn(tensor, model)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda is True


def test_output_fn_json(tensor):
    output = output_fn(tensor, JSON_CONTENT_TYPE)

    assert json.dumps(tensor.cpu().numpy().tolist()) in output
    assert JSON_CONTENT_TYPE in output


def test_output_fn_npy(tensor):
    output = output_fn(tensor, NPY_CONTENT_TYPE)

    stream = BytesIO()
    np.save(stream, tensor.cpu().numpy())

    assert stream.getvalue() in output
    assert NPY_CONTENT_TYPE in output


def test_output_fn_csv_long():
    tensor = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    output = output_fn(tensor, CSV_CONTENT_TYPE)

    assert '1,2,3\n4,5,6\n' in output
    assert CSV_CONTENT_TYPE in output


def test_output_fn_csv_float():
    tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    output = output_fn(tensor, CSV_CONTENT_TYPE)

    assert '1.0,2.0,3.0\n4.0,5.0,6.0\n' in output
    assert CSV_CONTENT_TYPE in output


def test_output_fn_bad_accept():
    with pytest.raises(UnsupportedAcceptTypeError):
        output_fn('', 'application/not_supported')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_output_fn_gpu():
    tensor_gpu = torch.LongTensor([[1, 2, 3], [4, 5, 6]]).cuda()

    output = output_fn(tensor_gpu, CSV_CONTENT_TYPE)

    assert '1,2,3\n4,5,6\n' in output
    assert CSV_CONTENT_TYPE in output


# TODO: add tests for transform_fn function when it's fixed in container_support
