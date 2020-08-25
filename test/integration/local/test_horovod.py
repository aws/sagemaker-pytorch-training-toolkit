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
def test_horovod_gpu(instances, processes, session, image_uri, framework_version, tmpdir):
    _test_horovod(1, 2, "local_gpu", session, image_uri, framework_version, tmpdir)


def _test_horovod(
    instances, processes, instance_type, session, image_uri, framework_version, tmpdir
):
    output_path = 'file://' + str(tmpdir)

    estimator = PyTorch(
        entry_point=os.path.join(resources_path, 'horovod', 'simple.py'),
        role='SageMakerRole',
        train_instance_type=instance_type,
        sagemaker_session=session,
        train_instance_count=instances,
        image_name=image_uri,
        output_path=output_path,
        framework_version=framework_version,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_network_interface_name': 'eth0',
                         'sagemaker_mpi_num_of_processes_per_host': processes})

    estimator.fit('file://{}'.format(training_dir))

    with tarfile.open(os.path.join(str(tmpdir), 'model.tar.gz')) as tar:
        tar.extractall(tmpdir)

    size = instances * processes

    for rank in range(size):
        local_rank = rank % processes
        filename = 'local-rank-%s-rank-%s' % (local_rank, rank)

        with open(os.path.join(str(tmpdir), filename)) as file:
            actual = json.load(file)
        expected = {'local-rank': local_rank, 'rank': rank, 'size': size}

        assert actual == expected
