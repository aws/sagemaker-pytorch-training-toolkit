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
import logging
from retrying import retry
import six
import socket
import sys
from sagemaker_training import entry_point, environment, errors, runner

MASTER_PORT = '7777'
LAUNCH_SMDATAPARALLEL_ENV_NAME = 'sagemaker_distributed_dataparallel_enabled'
LAUNCH_MPI_ENV_NAME = 'sagemaker_mpi_enabled'
LAUNCH_PYTORCH_DDP_ENV_NAME = "sagemaker_pytorch_ddp_enabled"

logger = logging.getLogger(__name__)


def train(training_environment):
    """Run PyTorch training on a user supplied module.

    The user supplied module is run in either a local or distributed SageMaker
    environment.

    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.
    if the environment contains multiple hosts, then a distributed learning
    task is started.

    Args:
        training_environment: training environment object containing environment
            variables, training arguments and hyperparameters.
    """
    # Block until all host DNS lookups succeed. Relies on retrying dns_lookup.
    logger.info('Block until all host DNS lookups succeed.')
    for host in training_environment.hosts:
        _dns_lookup(host)

    _set_nccl_environment(training_environment.network_interface_name)

    _set_distributed_environment(training_environment)

    mpi_enabled = training_environment.additional_framework_parameters.get(LAUNCH_MPI_ENV_NAME)

    pytorch_ddp_enabled = training_environment.additional_framework_parameters.get(
        LAUNCH_PYTORCH_DDP_ENV_NAME, False
    )

    smdataparallel_enabled = training_environment.additional_framework_parameters.get(
        LAUNCH_SMDATAPARALLEL_ENV_NAME, False
    )
    # default scenario
    runner_type = runner.ProcessRunnerType

    if training_environment.current_instance_group in training_environment.distribution_instance_groups:
        if mpi_enabled:
            runner_type = runner.MPIRunnerType
        elif pytorch_ddp_enabled:
            runner_type = runner.SMDataParallelRunnerType
            logger.info('Invoking SMDataParallel for native PT DDP job')
        elif smdataparallel_enabled:
            runner_type = runner.SMDataParallelRunnerType
            logger.info('Invoking SMDataParallel')
    logger.info('Invoking user training script.')
    try:
        entry_point.run(uri=training_environment.module_dir,
                        user_entry_point=training_environment.user_entry_point,
                        args=training_environment.to_cmd_args(),
                        env_vars=training_environment.to_env_vars(),
                        capture_error=True,
                        runner_type=runner_type)
    except errors.ExecuteUserScriptError as err:
        message = str(err)
        if message.find('terminate called after throwing an instance of \'gloo::EnforceNotMet\'') > -1:
            logger.warn('Known exception: {}'.format(message))
        else:
            info = sys.exc_info()
            six.reraise(info[0], err, info[2])


@retry(stop_max_delay=1000 * 60 * 15,
       wait_exponential_multiplier=100,
       wait_exponential_max=30000)
def _dns_lookup(host):
    """Retry DNS lookup on host."""
    return socket.gethostbyname(host)


def _set_distributed_environment(training_env):
    """Set environment variable for distributed training.

    Args:
        hosts: list of hosts that are used for training.
    """
    # According to https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    # hosts are sorted lexicographically.
    os.environ['MASTER_ADDR'] = training_env.master_hostname
    os.environ['MASTER_PORT'] = MASTER_PORT


def _set_nccl_environment(network_interface_name):
    """Set NCCL environment variables for the container.

    https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs

    Args:
        network_interface_name: The name of the network interface to use for
            distributed training.
    """
    # Set the network interface for inter node communication
    os.environ['NCCL_SOCKET_IFNAME'] = network_interface_name
    # Disable IB transport and force to use IP sockets by default
    os.environ['NCCL_IB_DISABLE'] = '1'
    # Set to INFO for more NCCL debugging information
    os.environ['NCCL_DEBUG'] = 'WARN'


def main():
    train(environment.Environment())
