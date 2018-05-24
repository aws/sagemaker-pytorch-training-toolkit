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
from sagemaker.pytorch.estimator import PyTorch


class PytorchTestEstimator(PyTorch):
    def __init__(self, docker_image_uri, **kwargs):
        super(PytorchTestEstimator, self).__init__(**kwargs)
        self.docker_image_uri = docker_image_uri

    def train_image(self):
        return self.docker_image_uri

    def create_model(self):
        model = super(PytorchTestEstimator, self).create_model()
        model.image = self.docker_image_uri
        return model
