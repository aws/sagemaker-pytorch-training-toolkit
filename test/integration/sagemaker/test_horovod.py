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

import json
import os
import tarfile

import pytest
from sagemaker.pytorch import PyTorch

from integration import resources_path, training_dir


@pytest.mark.skip_cpu
@pytest.mark.skip_generic
@pytest.mark.parametrize("instances, processes", [(1, 8), (2, 16)])
def test_horovod(instances, processes, sagemaker_session, image_uri, framework_version, tmpdir):
    default_bucket = sagemaker_session.default_bucket()
    output_path = "s3://" + os.path.join(default_bucket, "pytorch/horovod")

    estimator = PyTorch(
        entry_point=os.path.join(resources_path, 'horovod', 'simple.py'),
        role='SageMakerRole',
        train_instance_type="ml.p2.8xlarge",
        sagemaker_session=sagemaker_session,
        train_instance_count=instances,
        image_name=image_uri,
        output_path=output_path,
        framework_version=framework_version,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_num_of_processes_per_host': processes})

    input = sagemaker_session.upload_data(path=training_dir, key_prefix="pytorch/horovod")

    estimator.fit(input)

    bucket, key_prefix = estimator.model_data.replace("s3://", "").split("/", 1)
    sagemaker_session.download_data(
        path=str(tmpdir),
        bucket=bucket,
        key_prefix=key_prefix
    )

    with tarfile.open(os.path.join(str(tmpdir), 'model.tar.gz')) as tar:
        tar.extractall(tmpdir)

    size = instances * processes

    for rank in range(size):
        local_rank = rank % processes
        filename = 'local-rank-%s-rank-%s.json' % (local_rank, rank)

        with open(os.path.join(str(tmpdir), filename)) as file:
            actual = json.load(file)
        expected = {'local-rank': local_rank, 'rank': rank, 'size': size}

        assert actual == expected

