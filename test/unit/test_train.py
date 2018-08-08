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
import six
import shutil
import tempfile
import torch.nn as nn
from mock import MagicMock, PropertyMock
from mock import patch
from sagemaker_pytorch_container.training import main, train, _dns_lookup, MASTER_PORT
import sagemaker_containers.beta.framework as framework


@pytest.fixture(name='training_env')
def fixture_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1']
    env.network_interface_name = 'eth0'
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'model'))
    env.model_dir = os.path.join(tmp, 'model')
    env.module_name = 'user_script'
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


@patch('sagemaker_containers.beta.framework.modules.run_module')
@patch('socket.gethostbyname', MagicMock())
def test_train(run_module, training_env):
    train(training_env)

    run_module.assert_called_with(training_env.module_dir, training_env.to_cmd_args(),
                                          training_env.to_env_vars(), training_env.module_name)


@patch('sagemaker_containers.beta.framework.modules.run_module', MagicMock())
@patch('socket.gethostbyname', MagicMock())
def test_environment(training_env):
    train(training_env)

    # distributed training specific environment
    assert MASTER_PORT == os.environ['MASTER_PORT']
    assert training_env.hosts[0] == os.environ['MASTER_ADDR']

    # nccl specific environment
    assert training_env.network_interface_name == os.environ['NCCL_SOCKET_IFNAME']
    assert '1' == os.environ['NCCL_IB_DISABLE']
    assert 'WARN' == os.environ['NCCL_DEBUG']


@patch('sagemaker_pytorch_container.training.train')
@patch('sagemaker_containers.beta.framework.training_env')
def test_training_start(mock_training_env, mock_train, training_env):
    mock_training_env.return_value = training_env
    main()
    mock_train.assert_called_with(training_env)


@patch('socket.gethostbyname', MagicMock())
def test_train_with_missing_parameters(training_env, user_module):
    def user_module_train(missing_param):
        return nn.Module()

    user_module.train = user_module_train

    with pytest.raises(TypeError):
        train(user_module, training_env)


@patch('socket.gethostbyname', PropertyMock(return_value=True))
def test_dns_lookup_success():
    assert _dns_lookup('algo-1')


@patch('socket.gethostbyname', PropertyMock(return_value=False))
def test_dns_lookup_fail():
    assert not _dns_lookup('algo-1')


@patch('sagemaker_containers.beta.framework.modules.run_module')
@patch('socket.gethostbyname', MagicMock())
def test_gloo_exception_intercepted(run_module, training_env):
    output = 'terminate called after throwing an instance of \'gloo::EnforceNotMet\''
    run_module.side_effect = framework.errors.ExecuteUserScriptError(
        cmd='Command "/usr/bin/python -m userscript"',
        output=output.encode('latin1') if six.PY3 else output
    )
    train(training_env)
    run_module.assert_called()


@patch('sagemaker_containers.beta.framework.modules.run_module')
@patch('socket.gethostbyname', MagicMock())
def test_user_script_error_raised(run_module, training_env):
    output = 'Not \'gloo::EnforceNotMet\' exception.'
    run_module.side_effect = framework.errors.ExecuteUserScriptError(
        cmd='Command "/usr/bin/python -m userscript"',
        output=output.encode('latin1') if six.PY3 else output
    )
    with pytest.raises(framework.errors.ExecuteUserScriptError):
        train(training_env)
