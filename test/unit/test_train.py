import os
import pytest
import shutil
import tempfile
import torch
import torch.nn as nn
from mock import MagicMock
from pytorch_container.training import train, MODEL_FILE_NAME


@pytest.fixture()
def _training_env():
    env = MagicMock()
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

    train(_user_module_with_save, _training_env)

    assert _training_state.trained
    _user_module_with_save.save.assert_called_with(_training_state.model, _training_env.model_dir)


def test_train_with_default_save(_training_env, _user_module, _training_state):
    def user_module_train():
        _training_state.trained = True
        _training_state.model = nn.Module()
        return _training_state.model

    _user_module.train = user_module_train

    train(_user_module, _training_env)

    assert _training_state.trained
    assert os.path.exists(os.path.join(_training_env.model_dir, MODEL_FILE_NAME))


def test_default_save(_training_env, _user_module, _training_state):
    def user_module_train():
        _training_state.trained = True
        _training_state.model = nn.Module()
        return _training_state.model

    _user_module.train = user_module_train

    train(_user_module, _training_env)
    with open(os.path.join(_training_env.model_dir, MODEL_FILE_NAME), 'r') as f:
        try:
            the_model = nn.Module()
            the_model.load_state_dict(torch.load(f))
        except Exception, e:
            pytest.fail('Failed loading saved model. Exception: \'{}\''.format(e.message))


def test_train_with_all_parameters(_training_env, _user_module, _training_state):
    _training_env.training_parameters = {'first_param': 1, 'second_param': 2, 'third_param': 3}

    def user_module_train(first_param, second_param, third_param):
        _training_state.training_parameters = first_param, second_param, third_param
        return nn.Module()

    _user_module.train = user_module_train

    train(_user_module, _training_env)

    assert _training_state.training_parameters == (1, 2, 3)


def test_train_with_missing_parameters(_training_env, _user_module, _training_state):
    def user_module_train(missing_param):
        return nn.Module()

    _user_module.train = user_module_train

    with pytest.raises(TypeError):
        train(_user_module, _training_env)
