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
import shutil
import tempfile

import pytest
from sagemaker_training import errors, runner
import six
import torch.nn as nn
from mock import MagicMock, PropertyMock
from mock import patch

from sagemaker_pytorch_container.training import main, train, _dns_lookup, MASTER_PORT


@pytest.fixture(name='training_env')
def fixture_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1']
    env.master_hostname = 'algo-1'
    env.network_interface_name = 'eth0'
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'model'))
    env.model_dir = os.path.join(tmp, 'model')
    env.user_entry_point = 'user_script'
    env.current_instance_group = 'test1'
    env.distribution_instance_groups = ['test1']
    env.additional_framework_parameters = {}
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


@patch('sagemaker_training.entry_point.run')
@patch('socket.gethostbyname', MagicMock())
def test_train(run_entry_point, training_env):
    train(training_env)

    run_entry_point.assert_called_with(uri=training_env.module_dir,
                                       user_entry_point=training_env.user_entry_point,
                                       args=training_env.to_cmd_args(),
                                       env_vars=training_env.to_env_vars(),
                                       capture_error=True,
                                       runner_type=runner.ProcessRunnerType)


@patch("sagemaker_training.entry_point.run")
@patch('socket.gethostbyname', MagicMock())
def test_train_smdataparallel(run_module, training_env):
    training_env.additional_framework_parameters["sagemaker_distributed_dataparallel_enabled"] = True

    train(training_env)
    run_module.assert_called_with(
        uri=training_env.module_dir,
        user_entry_point=training_env.user_entry_point,
        args=training_env.to_cmd_args(),
        env_vars=training_env.to_env_vars(),
        capture_error=True,
        runner_type=runner.SMDataParallelRunnerType,
    )


@patch("sagemaker_training.entry_point.run")
@patch('socket.gethostbyname', MagicMock())
def test_train_pytorch_ddp(run_module, training_env):
    training_env.additional_framework_parameters["sagemaker_pytorch_ddp_enabled"] = True

    train(training_env)
    run_module.assert_called_with(
        uri=training_env.module_dir,
        user_entry_point=training_env.user_entry_point,
        args=training_env.to_cmd_args(),
        env_vars=training_env.to_env_vars(),
        capture_error=True,
        runner_type=runner.SMDataParallelRunnerType,
    )


@patch('sagemaker_training.entry_point.run', MagicMock())
@patch('socket.gethostbyname', MagicMock())
def test_environment(training_env):
    train(training_env)

    # distributed training specific environment
    assert MASTER_PORT == os.environ['MASTER_PORT']
    assert training_env.master_hostname == os.environ['MASTER_ADDR']

    # nccl specific environment
    assert training_env.network_interface_name == os.environ['NCCL_SOCKET_IFNAME']
    assert '1' == os.environ['NCCL_IB_DISABLE']
    assert 'WARN' == os.environ['NCCL_DEBUG']


@patch('sagemaker_pytorch_container.training.train')
@patch('sagemaker_training.environment.Environment')
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


@patch('sagemaker_training.entry_point.run')
@patch('socket.gethostbyname', MagicMock())
def test_gloo_exception_intercepted(run_entry_point, training_env):
    output = 'terminate called after throwing an instance of \'gloo::EnforceNotMet\''
    run_entry_point.side_effect = errors.ExecuteUserScriptError(
        cmd='Command "/usr/bin/python -m userscript"',
        output=output.encode('latin1') if six.PY3 else output
    )
    train(training_env)
    run_entry_point.assert_called()


@patch('sagemaker_training.entry_point.run')
@patch('socket.gethostbyname', MagicMock())
def test_user_script_error_raised(run_entry_point, training_env):
    output = 'Not \'gloo::EnforceNotMet\' exception.'
    run_entry_point.side_effect = errors.ExecuteUserScriptError(
        cmd='Command "/usr/bin/python -m userscript"',
        output=output.encode('latin1') if six.PY3 else output
    )
    with pytest.raises(errors.ExecuteUserScriptError):
        train(training_env)
