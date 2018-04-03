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
import torch
import os
import logging
import socket
import container_support as cs
from container_support.app import TrainingEngine

MODEL_FILE_NAME = 'model'
MASTER_PORT = '29500'

engine = TrainingEngine()
logger = logging.getLogger(__name__)


@engine.train()
def train(user_module, training_environment):
    """ Runs PyTorch training on a user supplied module in either a local or distributed
    SageMaker environment.
    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.
    if the environment contains multiple hosts, then a distributed learning
    task is started.
    Args:
        user_module : a user supplied module.
        training_environment : training environment object containing environment variables,
                               training arguments and hyperparameters
    """
    # Block until all host DNS lookups succeed. Relies on retrying dns_lookup.
    logger.info("Block until all host DNS lookups succeed.")
    for host in training_environment.hosts:
        _dns_lookup(host)

    sorted_hosts = sorted(training_environment.hosts)
    host_rank = sorted_hosts.index(training_environment.current_host)
    master_addr = sorted_hosts[0]

    # TODO (mvsusp): needs to be moved to container support package
    training_environment.training_parameters['host_rank'] = host_rank
    training_environment.training_parameters['master_addr'] = master_addr
    training_environment.training_parameters['master_port'] = MASTER_PORT

    model = user_module.train(**training_environment.training_parameters)

    if model:
        if hasattr(user_module, 'save'):
            logger.info("Using save function provided by the user.")
            user_module.save(model, training_environment.model_dir)
        elif training_environment.current_host == master_addr:
            _default_save(model, training_environment.model_dir)


def _default_save(model, model_dir):
    """Default logic to save a model to self.model_dir folder (/opt/ml/model).
    This function is called when a customer script does not provide a save() function.
        Args:
            model : module to save.
            model_dir : directory where module should be saved.
    """
    logger.info("Saving the model using default save function.")
    path = os.path.join(model_dir, MODEL_FILE_NAME)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)


@cs.retry(stop_max_delay=1000 * 60 * 15,
          wait_exponential_multiplier=100,
          wait_exponential_max=30000)
def _dns_lookup(host):
    """ Retrying dns lookup on host """
    return socket.gethostbyname(host)

