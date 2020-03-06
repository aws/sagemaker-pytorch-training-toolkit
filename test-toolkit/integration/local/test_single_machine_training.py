# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.pytorch import PyTorch

from utils.local_mode_utils import assert_files_exist
from integration import data_dir, mnist_script, ROLE


def test_mnist(ecr_image, processor, instance_type, sagemaker_local_session, tmpdir):
    estimator = PyTorch(entry_point=mnist_script,
                        role=ROLE,
                        image_name=ecr_image,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters={'processor': processor},
                        output_path='file://{}'.format(tmpdir))

    _train_and_assert_success(estimator, data_dir, str(tmpdir))


def _train_and_assert_success(estimator, input_dir, output_path):
    estimator.fit({'training': 'file://{}'.format(os.path.join(input_dir, 'training'))})

    success_files = {
        'model': ['model.pth'],
        'output': ['success'],
    }
    assert_files_exist(output_path, success_files)
