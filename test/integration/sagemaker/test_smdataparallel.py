# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from sagemaker.pytorch import PyTorch

from integration import resources_path, DEFAULT_TIMEOUT
from integration.sagemaker.timeout import timeout


@pytest.mark.skip_cpu
@pytest.mark.skip_generic
@pytest.mark.parametrize(
    "instances, train_instance_type",
    [(1, "ml.p3.16xlarge"), (2, "ml.p3.16xlarge"), (1, "ml.p3dn.24xlarge"), (2, "ml.p3dn.24xlarge")],
)
def test_smdataparallel_training(
    instances, train_instance_type, sagemaker_session, image_uri, framework_version, tmpdir
):
    default_bucket = sagemaker_session.default_bucket()
    output_path = "s3://" + os.path.join(default_bucket, "pytorch/smdataparallel")

    estimator = PyTorch(
        entry_point=os.path.join(resources_path, "mnist", "smdataparallel_mnist.py"),
        role="SageMakerRole",
        train_instance_type=train_instance_type,
        sagemaker_session=sagemaker_session,
        train_instance_count=instances,
        image_name=image_uri,
        output_path=output_path,
        framework_version=framework_version,
        hyperparameters={
            "sagemaker_distributed_dataparallel_enabled": True
        }
    )

    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator.fit()
