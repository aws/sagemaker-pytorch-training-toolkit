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
import pytest
import shutil
import tempfile
import torch
import torch.nn as nn
from mock import MagicMock
from mock import patch
from pytorch_container.training import train, MODEL_FILE_NAME, _default_save, _dns_lookup, MASTER_PORT


@pytest.fixture(name='training_env')
def fixture_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1']
    env.network_interface_name = 'eth0'
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'model'))
    env.model_dir = os.path.join(tmp, 'model')
    yield env
    shutil.rmtree(tmp)


@pytest.fixture(name='training_state')
def fixture_training_state():
    training_state = MagicMock()
    training_state.trained = False
    return training_state


@pytest.fixture(name='user_module')
def fixture_user_module():
    return MagicMock(spec=['train'])


@pytest.fixture(name='user_module_with_save')
def fixture_user_module_with_save():
    return MagicMock(spec=['train', 'save'])


def test_train(training_env, user_module_with_save, training_state):
    def user_module_train():
        training_state.trained = True
        training_state.model = nn.Module()
        return training_state.model

    user_module_with_save.train = user_module_train

    with patch('socket.gethostbyname'):
        train(user_module_with_save, training_env)

    assert training_state.trained
    user_module_with_save.save.assert_called_with(training_state.model, training_env.model_dir)


def test_train_with_default_save(training_env, user_module, training_state):
    def user_module_train():
        training_state.trained = True
        training_state.model = nn.Module()
        return training_state.model

    user_module.train = user_module_train

    with patch('socket.gethostbyname'):
        train(user_module, training_env)

    assert training_state.trained
    assert os.path.exists(os.path.join(training_env.model_dir, MODEL_FILE_NAME))


def test_default_save(training_env):
    model = nn.Module()
    _default_save(model, training_env.model_dir)
    f = os.path.join(training_env.model_dir, MODEL_FILE_NAME)
    try:
        the_model = nn.Module()
        the_model.load_state_dict(torch.load(f))
    except Exception as e:
            pytest.fail('Failed loading saved model. Exception: \'{}\''.format(e))


def test_train_with_all_parameters(training_env, user_module, training_state):
    training_env.training_parameters = {'first_param': 1, 'second_param': 2}

    def user_module_train(host_rank, master_addr, master_port, first_param, second_param):
        training_state.training_parameters = host_rank, master_addr, master_port, first_param, second_param
        return nn.Module()

    user_module.train = user_module_train

    with patch('socket.gethostbyname'):
        train(user_module, training_env)

    assert training_state.training_parameters == (0, 'algo-1', MASTER_PORT, 1, 2)


def test_train_with_missing_parameters(training_env, user_module, training_state):
    def user_module_train(missing_param):
        return nn.Module()

    user_module.train = user_module_train

    with patch('socket.gethostbyname'), \
            pytest.raises(TypeError):
        train(user_module, training_env)


def test_dns_lookup_success():
    with patch('socket.gethostbyname') as mock_gethostbyname:
        mock_gethostbyname.return_value = True
        assert _dns_lookup('algo-1')


def test_dns_lookup_fail():
    with patch('socket.gethostbyname') as mock_gethostbyname:
        mock_gethostbyname.return_value = False
        assert not _dns_lookup('algo-1')
