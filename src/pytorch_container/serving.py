import json
import os
import pickle
import numpy as np

from container_support.app import ServingEngine
from container_support.serving import JSON_CONTENT_TYPE, UnsupportedContentTypeError, UnsupportedAcceptTypeError

from ml_framework import PICKLE_CONTENT_TYPE

engine = ServingEngine()


@engine.model_fn()
def model_fn(model_dir):
    with open(os.path.join(model_dir, 'model.pickle'), 'r') as f:
        return pickle.load(f)


@engine.input_fn()
def input_fn(serialized_input_data, content_type):
    """A default input fn to handle JSON and PICKLE formats. The required formats for a SageMaker Container are
    JSON and CSV.

    Args:
        serialized_input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: deserialized input_data
    """
    if content_type == JSON_CONTENT_TYPE:
        return json.loads(serialized_input_data)

    if content_type == PICKLE_CONTENT_TYPE:
        return pickle.loads(serialized_input_data)

    raise UnsupportedContentTypeError(content_type)


@engine.predict_fn()
def predict_fn(input_data, model):
    """A default prediction function for a Generic Framework.

    Args:
        input_data: input data for prediction deserialized by input_fn
        model: model loaded in memory by model_fn

    Returns: a prediction
    """
    pass


@engine.output_fn()
def output_fn(prediction_output, accept):
    """A default output_fn for a Generic Framework

    Args:
        prediction_output: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns
        output data serialized
    """
    prediction_output = prediction_output.tolist() if hasattr(prediction_output, 'tolist') else prediction_output

    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), JSON_CONTENT_TYPE

    if accept == PICKLE_CONTENT_TYPE:
        return pickle.dumps(prediction_output), PICKLE_CONTENT_TYPE

    raise UnsupportedAcceptTypeError(accept)


@engine.transform_fn()
def transform_fn(model, data, content_type, accept):
    input_data = input_fn(data, content_type)
    prediction = predict_fn(input_data, model)
    output_data, accept = output_fn(prediction, accept)
    return output_data, accept


@engine.load_dependencies()
def load_dependencies():
    """This function is only called once by the container support before it starts the Flask servers, it useful to load
    framework specific dependencies
    """
    pass
