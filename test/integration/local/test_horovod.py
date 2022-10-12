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

from integration import resources_path


@pytest.mark.skip_cpu
@pytest.mark.skip_generic
def test_horovod_simple(sagemaker_local_session, image_uri, framework_version, tmpdir):
    instances, processes = 1, 2
    output_path = 'file://' + str(tmpdir)

    estimator = PyTorch(
        entry_point=os.path.join(resources_path, 'horovod', 'simple.py'),
        role='SageMakerRole',
        train_instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        train_instance_count=instances,
        image_name=image_uri,
        output_path=output_path,
        framework_version=framework_version,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_num_of_processes_per_host': processes})

    estimator.fit()

    with tarfile.open(os.path.join(str(tmpdir), 'model.tar.gz')) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, tmpdir)

    size = instances * processes

    for rank in range(size):
        local_rank = rank % processes
        # The simple.py script should create a JSON file with this name
        filename = 'local-rank-%s-rank-%s.json' % (local_rank, rank)

        with open(os.path.join(str(tmpdir), filename)) as file:
            actual = json.load(file)
        expected = {'local-rank': local_rank, 'rank': rank, 'size': size}

        assert actual == expected


@pytest.mark.skip(reason="Temporarily skip for 1.6.0")
@pytest.mark.skip_cpu
@pytest.mark.skip_generic
def test_horovod_training(sagemaker_local_session, image_uri, framework_version, tmpdir):
    estimator = PyTorch(
        entry_point=os.path.join(resources_path, 'horovod', 'train.py'),
        role='SageMakerRole',
        train_instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        train_instance_count=1,
        image_name=image_uri,
        framework_version=framework_version,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_num_of_processes_per_host': 2,
                         'epochs': 1})

    estimator.fit()
