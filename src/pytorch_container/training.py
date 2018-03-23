import torch
import os
import logging
import socket
import container_support as cs
from container_support.app import TrainingEngine

MODEL_FILE_NAME = 'model'

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
        dns_lookup(host)

    rank = sorted(training_environment.hosts).index(training_environment.current_host)
    # TODO: should world size be something different?
    world_size = len(training_environment.hosts)

    training_environment.training_parameters['rank'] = rank
    training_environment.training_parameters['world_size'] = world_size

    model = user_module.train(**training_environment.training_parameters)

    if model:
        if hasattr(user_module, 'save'):
            logger.info("Using save function provided by the user.")
            user_module.save(model, training_environment.model_dir)
        else:
            _default_save(model, training_environment.model_dir, rank)


def _default_save(model, model_dir, rank):
    """Default logic to save a model to self.model_dir folder (/opt/ml/model),
    will save the model only if current host has rank=0.
    This function is called when a customer script does not provide a save() function.
        Args:
            model : module to save.
            model_dir : directory where module should be saved.
    """
    if rank == 0:
        logger.info("Saving the model using default save function.")
        path = os.path.join(model_dir, MODEL_FILE_NAME)
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save(model.state_dict(), path)


# TODO: needs to be moved to container support package
@cs.retry(stop_max_delay=1000 * 60 * 15,
          wait_exponential_multiplier=100,
          wait_exponential_max=30000)
def dns_lookup(host):
    """ Retrying dns lookup on host """
    return socket.gethostbyname(host)

