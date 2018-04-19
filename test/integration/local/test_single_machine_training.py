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
from test.utils import local_mode
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources'))
mnist_path = os.path.join(resources_path, 'mnist')
data_dir = os.path.join(mnist_path, 'data')

training_dir = os.path.join(data_dir, 'training')

mnist_script = 'mnist.py'

ENTRYPOINT = ["python", "-m", "pytorch_container.start"]


def test_mnist_cpu(docker_image, opt_ml, use_gpu):
    local_mode.train(mnist_script, data_dir, docker_image, opt_ml,
                     source_dir=mnist_path, use_gpu=use_gpu, entrypoint=ENTRYPOINT)

    assert local_mode.file_exists(opt_ml, 'model/model'), 'Model file was not created'
    assert local_mode.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'
