from __future__ import absolute_import
import json
import logging
import numpy as np
from six import StringIO, BytesIO
import torch
from torch.autograd import Variable

from container_support.app import ServingEngine
from container_support.serving import JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, NPY_CONTENT_TYPE, \
    UnsupportedContentTypeError, UnsupportedAcceptTypeError

engine = ServingEngine()
logger = logging.getLogger(__name__)


@engine.model_fn()
def model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model cannot be provided.
    Users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """
    raise NotImplementedError('No default model_fn provided. User should provide model_fn in script.')


@engine.input_fn()
def input_fn(serialized_input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.
    Args:
        serialized_input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    input_data = _deserialize_input(serialized_input_data, content_type)
    return torch.cuda.FloatTensor(input_data) if torch.cuda.is_available() else torch.FloatTensor(input_data)


@engine.predict_fn()
def predict_fn(input_data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        input_data: input data (torch.FloatTensor) for prediction deserialized by input_fn
        model: PyTvorch model loaded in memory by model_fn

    Returns: a prediction
    """
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()
    model.eval()
    output = model(Variable(input_data))
    return output


@engine.output_fn()
def output_fn(prediction_output, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPZ format.

    Args:
        prediction_output: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns
        output data serialized
    """
    if type(prediction_output) == Variable:
        prediction_output = prediction_output.data

    return _serialize_output(prediction_output, accept), accept


# TODO: this function is actually never called:
#       https://github.com/aws/sagemaker-container-support/blob/mvs-poc/src/container_support/app.py#L110-L116
@engine.transform_fn()
def transform_fn(model, data, content_type, accept):
    raise NotImplementedError('transform_fn is never called in framework container.')
    input_data = input_fn(data, content_type, model)
    prediction = predict_fn(input_data, model)
    output_data, accept = output_fn(prediction, accept)
    return output_data, accept


def _deserialize_input(serialized_input_data, content_type):
    # TODO: Move deserialization of serialized_input_data to np.array to conatiner_support
    #       in order for it to be reused in all or some other containers
    if content_type == JSON_CONTENT_TYPE:
        return np.array(json.loads(serialized_input_data), dtype=np.float32)

    if content_type == CSV_CONTENT_TYPE:
        return np.genfromtxt(StringIO(serialized_input_data), filling_values=0, dtype=np.float32, delimiter=',')

    if content_type == NPY_CONTENT_TYPE:
        return np.load(BytesIO(serialized_input_data))

    raise UnsupportedContentTypeError(content_type)


def _serialize_output(prediction_output, content_type):
    # TODO: Move serialization of prediction from np.array to conatiner_support
    #       in order for it to be reused in all or some other containers
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

    raise UnsupportedAcceptTypeError(content_type)
