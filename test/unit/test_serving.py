import pytest
import json
import numpy as np
import os
import csv
from six import StringIO
import torch
from torch.autograd import Variable

from container_support.serving import JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, \
    UnsupportedContentTypeError, UnsupportedAcceptTypeError

from pytorch_container.serving import model_fn, input_fn, predict_fn, output_fn, transform_fn


@pytest.fixture(scope='session', name='tensor')
def _tensor():
    return torch.rand(5, 10, 7, 9)


def test_model_fn():
    with pytest.raises(NotImplementedError):
        model_fn('model_dir')


def test_input_fn_json(tensor):
    json_data = json.dumps(tensor.numpy().tolist())
    deserialized_np_array = input_fn(json_data, JSON_CONTENT_TYPE)

    assert torch.equal(tensor, deserialized_np_array)
    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert deserialized_np_array.is_cuda == torch.cuda.is_available()


def test_input_fn_csv():
    array = [[1, 2, 3], [4, 5, 6]]
    str_io = StringIO()
    csv.writer(str_io, delimiter=',').writerows(array)

    deserialized_np_array = input_fn(str_io.getvalue(), CSV_CONTENT_TYPE)

    tensor = torch.FloatTensor(array)
    assert type(deserialized_np_array) == torch.FloatTensor
    assert torch.equal(tensor, deserialized_np_array)
    assert deserialized_np_array.is_cuda == torch.cuda.is_available()


def test_input_fn_csv_bad_columns():
    str_io = StringIO()
    csv_writer = csv.writer(str_io, delimiter=',')
    csv_writer.writerow([1, 2, 3])
    csv_writer.writerow([1, 2, 3, 4])

    with pytest.raises(ValueError):
        input_fn(str_io.getvalue(), CSV_CONTENT_TYPE)


def test_input_fn_bad_content_type():
    with pytest.raises(UnsupportedContentTypeError):
        input_fn('', 'application/not_supported')


def test_predict_fn(tensor):
    pass
    #predicted_data = predict_fn(np_array, FakeModel())
    #assert np.array_equal(fake_predict(np_array), predicted_data)


def test_output_fn_json(tensor):
    output = output_fn(tensor, JSON_CONTENT_TYPE)

    assert json.dumps(tensor.numpy().tolist()) in output
    assert JSON_CONTENT_TYPE in output


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
