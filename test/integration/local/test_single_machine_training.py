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
import os

from test.utils import local_mode
from test.integration import data_dir, fastai_path, fastai_mnist_script, mnist_script


def test_mnist(docker_image, opt_ml, use_gpu, processor):
    local_mode.train(mnist_script, data_dir, docker_image, opt_ml, use_gpu=use_gpu,
                     hyperparameters={'processor': processor})

    assert local_mode.file_exists(opt_ml, 'model/model.pth'), 'Model file was not created'
    assert local_mode.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_fastai_mnist(docker_image, opt_ml, use_gpu, py_version):
    if py_version != 'py3':
        print('Skipping the test because fastai supports >= Python 3.6.')
        return

    local_mode.train(fastai_mnist_script, os.path.join(fastai_path, 'mnist_tiny'), docker_image,
                     opt_ml, use_gpu=use_gpu)

    assert local_mode.file_exists(opt_ml, 'model/model.pth'), 'Model file was not created'
    assert local_mode.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'
