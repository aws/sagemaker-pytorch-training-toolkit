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
import os
from os.path import join
import logging
import platform
import pytest
import shutil
import sys
import tempfile


from test.utils import local_mode

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)


dir_path = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption('--build-image', '-D', action="store_true")
    parser.addoption('--build-base-image', '-B', action="store_true")
    parser.addoption('--install-container-support', '-C', action="store_true")
    parser.addoption('--docker-base-name', default='pytorch')
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--framework-version', default='0.3.1')
    parser.addoption('--py-version', choices=['2', '3'], default='2')
    parser.addoption('--processor', choices=['gpu','cpu'], default='cpu')
    # If not specified, will default to {framework-version}-{processor}-py{py-version}
    parser.addoption('--tag', default=None)


@pytest.fixture(scope='session')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session')
def py_version(request):
    return 'py{}'.format(int(request.config.getoption('--py-version')))


@pytest.fixture(scope='session')
def processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='session')
def tag(request, framework_version, processor, py_version):
    provided_tag = request.config.getoption('--tag')
    default_tag = '{}-{}-{}'.format(framework_version, processor, py_version)
    return provided_tag if provided_tag else default_tag


@pytest.fixture(scope='session')
def docker_image(docker_base_name, tag):
    return '{}:{}'.format(docker_base_name, tag)


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield opt_ml_dir

    shutil.rmtree(tmp, True)


@pytest.fixture(scope='session')
def use_gpu(processor):
    return processor == 'gpu'


@pytest.fixture(scope='session', autouse=True)
def install_container_support(request):
    install = request.config.getoption('--install-container-support')
    if install:
        local_mode.install_container_support()


@pytest.fixture(scope='session', autouse=True)
def build_base_image(request, framework_version, py_version, processor, tag, docker_base_name):
    build_base_image = request.config.getoption('--build-base-image')
    if build_base_image:
        return local_mode.build_base_image(framework_name=docker_base_name,
                                           framework_version=framework_version,
                                           py_version=py_version,
                                           base_image_tag=tag,
                                           processor=processor,
                                           cwd=os.path.join(dir_path, '..'))

    return tag


@pytest.fixture(scope='session', autouse=True)
def build_image(request, framework_version, py_version, processor, tag, docker_base_name):
    build_image = request.config.getoption('--build-image')
    if build_image:
        return local_mode.build_image(framework_name=docker_base_name,
                                      framework_version=framework_version,
                                      py_version=py_version,
                                      processor=processor,
                                      tag=tag,
                                      cwd=os.path.join(dir_path, '..'))

    return tag
