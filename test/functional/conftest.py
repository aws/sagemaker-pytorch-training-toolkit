import os
from os.path import join
import platform
import pytest
import shutil
import tempfile
import utils

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield opt_ml_dir

    shutil.rmtree(tmp, True)


@pytest.fixture(scope='session', params=["0.3.1"])
def image_name(request):
    def get_image_name(framework_version=request.param, device='cpu', py_version='py2'):
        build_image = request.config.getoption('--dont-build')
        if build_image:
            return utils.build_image(framework_version, device, py_version, cwd=join(dir_path, '..', '..'))
        return utils.get_image_tag(framework_version, device, py_version)
    return get_image_name
