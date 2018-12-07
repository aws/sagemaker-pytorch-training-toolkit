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

import logging

import torch
from sagemaker_containers.beta.framework import (content_types, encoders, env, modules, transformer,
                                                 worker)

logger = logging.getLogger(__name__)


def default_model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model cannot be provided.
    Users should provide customized model_fn() in script.

    Args:
        model_dir: a directory where model is saved.

    Returns: A PyTorch model.
    """
    return transformer.default_model_fn(model_dir)


def default_input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np_array = encoders.decode(input_data, content_type)
    tensor = torch.FloatTensor(
        np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    return tensor.to(device)


def default_predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.

    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTvorch model loaded in memory by model_fn

    Returns: a prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output


def default_output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPZ format.

    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns: output data serialized
    """
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy()

    return worker.Response(response=encoders.encode(prediction, accept), mimetype=accept)


def _user_module_transformer(user_module):
    model_fn = getattr(user_module, 'model_fn', default_model_fn)
    input_fn = getattr(user_module, 'input_fn', default_input_fn)
    predict_fn = getattr(user_module, 'predict_fn', default_predict_fn)
    output_fn = getattr(user_module, 'output_fn', default_output_fn)

    return transformer.Transformer(model_fn=model_fn, input_fn=input_fn, predict_fn=predict_fn,
                                   output_fn=output_fn)


app = None


def main(environ, start_response):
    global app
    if app is None:
        serving_env = env.ServingEnv()
        user_module = modules.import_module(serving_env.module_dir, serving_env.module_name)

        user_module_transformer = _user_module_transformer(user_module)

        user_module_transformer.initialize()

        app = worker.Worker(transform_fn=user_module_transformer.transform,
                            module_name=serving_env.module_name)

    return app(environ, start_response)
