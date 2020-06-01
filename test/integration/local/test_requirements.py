# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.pytorch import PyTorch

from utils.local_mode_utils import assert_files_exist
from integration import requirements_dir, requirements_script, ROLE


def test_requirements_file(image_uri, instance_type, sagemaker_local_session, tmpdir):
    estimator = PyTorch(
        entry_point=requirements_script,
        source_dir=requirements_dir,
        role=ROLE,
        image_name=image_uri,
        train_instance_count=1,
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_local_session,
        output_path='file://{}'.format(tmpdir)
    )

    estimator.fit()

    success_files = {'output': ['success']}
    assert_files_exist(str(tmpdir), success_files)
