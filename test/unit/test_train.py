import os
import pytest
import shutil
import tempfile
import torch
import torch.nn as nn
from mock import MagicMock
from mock import patch
from pytorch_container.training import train, MODEL_FILE_NAME, _default_save, _dns_lookup, MASTER_PORT


@pytest.fixture()
def _training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1']
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'model'))
    env.model_dir = os.path.join(tmp, 'model')
    yield env
    shutil.rmtree(tmp)


@pytest.fixture()
def _training_state():
    training_state = MagicMock()
    training_state.trained = False
    return training_state


@pytest.fixture()
def _user_module():
    return MagicMock(spec=['train'])


@pytest.fixture()
def _user_module_with_save():
    return MagicMock(spec=['train', 'save'])


def test_train(_training_env, _user_module_with_save, _training_state):
    def user_module_train():
        _training_state.trained = True
        _training_state.model = nn.Module()
        return _training_state.model

    _user_module_with_save.train = user_module_train

    with patch('socket.gethostbyname'):
        train(_user_module_with_save, _training_env)

    assert _training_state.trained
    _user_module_with_save.save.assert_called_with(_training_state.model, _training_env.model_dir)


def test_train_with_default_save(_training_env, _user_module, _training_state):
    def user_module_train():
        _training_state.trained = True
        _training_state.model = nn.Module()
        return _training_state.model

    _user_module.train = user_module_train

    with patch('socket.gethostbyname'):
        train(_user_module, _training_env)

    assert _training_state.trained
    assert os.path.exists(os.path.join(_training_env.model_dir, MODEL_FILE_NAME))


def test_default_save(_training_env):
    model = nn.Module()
    _default_save(model, _training_env.model_dir)
    f = os.path.join(_training_env.model_dir, MODEL_FILE_NAME)
    try:
        the_model = nn.Module()
        the_model.load_state_dict(torch.load(f))
    except Exception as e:
            pytest.fail('Failed loading saved model. Exception: \'{}\''.format(e.message))


def test_train_with_all_parameters(_training_env, _user_module, _training_state):
    _training_env.training_parameters = {'first_param': 1, 'second_param': 2}

    def user_module_train(host_rank, master_addr, master_port, first_param, second_param):
        _training_state.training_parameters = host_rank, master_addr, master_port, first_param, second_param
        return nn.Module()

    _user_module.train = user_module_train

    with patch('socket.gethostbyname'):
        train(_user_module, _training_env)

    assert _training_state.training_parameters == (0, 'algo-1', MASTER_PORT, 1, 2)


def test_train_with_missing_parameters(_training_env, _user_module, _training_state):
    def user_module_train(missing_param):
        return nn.Module()

    _user_module.train = user_module_train

    with patch('socket.gethostbyname'), \
            pytest.raises(TypeError):
        train(_user_module, _training_env)


def test_dns_lookup_success():
    with patch('socket.gethostbyname') as mock_gethostbyname:
        mock_gethostbyname.return_value = True
        assert _dns_lookup('algo-1')


def test_dns_lookup_fail():
    with patch('socket.gethostbyname') as mock_gethostbyname:
        mock_gethostbyname.return_value = False
        assert not _dns_lookup('algo-1')
