# TODO: use proper local mode when available
import json
import logging
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
from time import sleep
import torch

import boto3
import os
import requests
import yaml

from botocore.exceptions import ClientError
from sagemaker import fw_utils


CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'

REQUEST_URL = "http://localhost:8080/invocations"
JSON_CONTENT_TYPE = "application/json"


CONTAINER_PREFIX = "algo"
DOCKER_COMPOSE_FILENAME = 'docker-compose.yaml'
SAGEMAKER_REGION = 'us-west-2'
DEFAULT_HYPERPARAMETERS = {
    'sagemaker_enable_cloudwatch_metrics': False,
    'sagemaker_container_log_level': str(logging.INFO),
    'sagemaker_region': SAGEMAKER_REGION
}
DEFAULT_HOSTING_ENV = [
    'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS=false',
    'SAGEMAKER_CONTAINER_LOG_LEVEL={}'.format(logging.DEBUG),
    'SAGEMAKER_REGION={}'.format(SAGEMAKER_REGION)
]
ENTRYPOINT = ["python", "-m", "pytorch_container.start"]

def build_image(framework_version, device_type='cpu', py_version='py2', cwd='.'):
    image_tag = get_image_tag(framework_version, device_type, py_version)

    dockerfile_location = os.path.join('docker', framework_version, 'final', py_version,
                                       'Dockerfile.{}'.format(device_type))

    subprocess.check_call(['docker', 'build', '-t', image_tag, '-f', dockerfile_location, '.'], cwd=cwd)
    print('created image {}'.format(image_tag))
    return image_tag


def get_image_tag(framework_version, device_type='cpu', py_version='py2'):
    return 'sagemaker-pytorch:{}-{}-{}'.format(framework_version, device_type, py_version)


def serialize_hyperparameters(hp):
    return {str(k): json.dumps(v) for (k, v) in hp.items()}


def save_as_json(data, filename):
    with open(filename, "wt") as f:
        json.dump(data, f)


def train(customer_script, data_dir, image_name, opt_ml, cluster_size=1, hyperparameters={}, additional_volumes=[],
          additional_env_vars=[], use_gpu=False, entrypoint=ENTRYPOINT, source_dir=None):
    print("training")
    tmpdir = create_training(data_dir, customer_script, opt_ml, image_name, additional_volumes, additional_env_vars,
                             hyperparameters, cluster_size, entrypoint=entrypoint, source_dir=source_dir)
    command = create_docker_command(tmpdir, use_gpu)
    start_docker(tmpdir, command)
    purge()


def purge():
    """
    Kills all running containers whose names match those in the cluster. No
    validation done if the containers are actually running.
    :param cluster_size: the size of the cluster, used to determine the names
    """
    chain_docker_cmds('docker ps -q', 'docker rm -f')
    chain_docker_cmds('docker images -f dangling=true -q', 'docker rmi -f')
    chain_docker_cmds('docker network ls -q', 'docker network rm')


def chain_docker_cmds(cmd, cmd2):
    docker_tags = subprocess.check_output(cmd.split(' ')).split('\n')

    if any(docker_tags):
        try:
            subprocess.check_output(cmd2.split(' ') + docker_tags, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            pass


class Container(object):
    def __init__(self, tmpdir, command, startup_delay=5):
        self.command = command
        self.compose_file = os.path.join(tmpdir, DOCKER_COMPOSE_FILENAME)
        self.startup_delay = startup_delay
        self._process = None

    def __enter__(self):
        self._process = subprocess.Popen(self.command)
        sleep(self.startup_delay)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._process.terminate()
        purge()


def start_docker(tmpdir, command):
    compose_file = os.path.join(tmpdir, DOCKER_COMPOSE_FILENAME)

    try:
        subprocess.check_call(command)
    finally:
        shutdown(compose_file)


def shutdown(compose_file):
    print("shutting down")
    subprocess.call(['docker-compose', '-f', compose_file, 'down'])


def create_docker_command(tmpdir, use_gpu=False, detached=False):
    compose_cmd = 'nvidia-docker-compose' if use_gpu else 'docker-compose'

    command = [
        compose_cmd,
        '-f',
        os.path.join(tmpdir, DOCKER_COMPOSE_FILENAME),
        'up',
        '--build'
    ]

    if detached:
        command.append('-d')

    print('docker command: {}'.format(' '.join(command)))
    return command


def create_training(data_dir, customer_script, optml, image, additional_volumes, additional_env_vars,
                    additional_hps={}, cluster_size=1, source_dir=None, entrypoint=None):
    session = boto3.Session()
    tmpdir = os.path.abspath(optml)

    hosts = create_host_names(cluster_size)
    print('creating hosts: {}'.format(hosts))

    config = create_input_data_config(data_dir)
    hyperparameters = read_hyperparameters(additional_hps)

    if customer_script:
        s3_script_path = fw_utils.tar_and_upload_dir(session=session,
                                                     bucket=default_bucket(session),
                                                     s3_key_prefix='test',
                                                     script=customer_script,
                                                     directory=source_dir)[0]
        hyperparameters.update({
            'sagemaker_submit_directory': s3_script_path,
            'sagemaker_program': os.path.basename(customer_script)
        })

    for host in hosts:
        for d in ['input', 'input/config', 'output', 'model']:
            os.makedirs(os.path.join(tmpdir, host, d))

        write_hyperparameters(tmpdir, host, hyperparameters)
        write_resource_config(tmpdir, hosts, host)
        write_inputdataconfig(tmpdir, host, config)

        shutil.copytree(data_dir, os.path.join(tmpdir, host, 'input', 'data'))

    write_docker_file('train', tmpdir, hosts, image, additional_volumes, additional_env_vars, customer_script,
                      source_dir, entrypoint)

    print("training dir: \n{}".format(str(subprocess.check_output(['ls', '-lR', tmpdir]).decode('utf-8'))))

    return tmpdir


def write_inputdataconfig(path, current_host, inputdataconfig):
    filename = os.path.join(path, current_host, 'input', 'config', 'inputdataconfig.json')
    write_json_file(filename, inputdataconfig)


def write_hyperparameters(path, current_host, hyperparameters):
    serialized = {k: json.dumps(v) for k, v in hyperparameters.items()}
    filename = os.path.join(path, current_host, 'input', 'config', 'hyperparameters.json')
    write_json_file(filename, serialized)


def read_hyperparameters(additonal_hyperparameters={}):
    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    hyperparameters.update(additonal_hyperparameters)

    print('hyperparameters: {}'.format(hyperparameters))
    return hyperparameters


def create_input_data_config(data_path):
    channels = []
    for (root, dirs, files) in os.walk(data_path):
        channels.extend(dirs)
        del dirs

    config = {c: {'ContentType': 'application/octet-stream'} for c in channels}
    print('input data config: {}'.format(config))
    return config


def write_docker_file(command, tmpdir, hosts, image, additional_volumes, additional_env_vars, customer_script,
                      source_dir, entrypoint):
    filename = os.path.join(tmpdir, DOCKER_COMPOSE_FILENAME)
    content = create_docker_compose(command, tmpdir, hosts, image, additional_volumes, additional_env_vars,
                                    customer_script, source_dir, entrypoint)

    print('docker compose file: \n{}'.format(content))
    with open(filename, 'w') as f:
        f.write(content)


def create_docker_services(command, tmpdir, hosts, image, additional_volumes, additional_env_vars, customer_script,
                           source_dir, entrypoint):
    environment = []
    session = boto3.Session()

    optml_dirs = set()
    if command == 'train':
        optml_dirs = {'output', 'input'}

    elif command == 'serve':
        environment.extend(DEFAULT_HOSTING_ENV)

        if customer_script:
            s3_script_path = fw_utils.tar_and_upload_dir(session=session,
                                                         bucket=default_bucket(session),
                                                         s3_key_prefix='test',
                                                         script=customer_script,
                                                         directory=source_dir)[0]

            environment.extend([
                'SAGEMAKER_PROGRAM={}'.format(os.path.basename(customer_script)),
                'SAGEMAKER_SUBMIT_DIRECTORY={}'.format(s3_script_path)
            ])
    else:
        raise ValueError('Unexpected command: {}'.format(command))

    environment.extend(credentials_to_env(session))

    environment.extend(additional_env_vars)

    return {h: create_docker_host(tmpdir, h, image, environment, optml_dirs, command, additional_volumes, entrypoint)
            for h in
            hosts}


def create_docker_host(tmpdir, host, image, environment, optml_subdirs, command, volumes, entrypoint=None):
    optml_volumes = optml_volumes_list(tmpdir, host, optml_subdirs)
    optml_volumes = ['/private' + v if v.startswith('/var') else v for v in optml_volumes]
    optml_volumes.extend(volumes)

    host_config = {
        'image': image,
        'stdin_open': True,
        'tty': True,
        'volumes': optml_volumes,
        'environment': environment,
        'command': command,
    }

    if entrypoint:
        host_config['entrypoint'] = entrypoint

    if command == 'serve':
        host_config.update({
            'ports': [
                '8080:8080'
            ]
        })

    return host_config


def optml_volumes_list(opt_root_folder, host, subdirs, single_model_dir=False):
    """
    It takes a folder with the necessary files for training and creates a list of opt volumes that
    the Container needs to start.
    If args.single_model_dir is True, all the hosts will point the opt/ml/model subdir to the first container. That is
    useful for distributed training, so all the containers can read and write the same checkpoints.

    :param opt_root_folder: root folder with the contents to be mapped to the container
    :param host: host name of the container
    :param subdirs: list of subdirs that will be mapped. Example: ['input', 'output', 'model']
    :param args: command line arguments
    :return:
    """
    volumes_map = []

    # If it is single mode dir we want to map the same model dir and share between hosts
    if single_model_dir:
        volumes_map.append('{}:/opt/ml/model'.format(os.path.join(opt_root_folder, 'algo-1/model')))
    else:
        # else we want to add model to the list of subdirs so it will be created for each container.
        subdirs.add('model')

    for subdir in subdirs:
        volume_root = os.path.join(opt_root_folder, host, subdir)
        volumes_map.append('{}:/opt/ml/{}'.format(volume_root, subdir))

    return volumes_map


def credentials_to_env(session):
    try:
        creds = session.get_credentials()
        access_key = creds.access_key
        secret_key = creds.secret_key

        return [
            'AWS_ACCESS_KEY_ID=%s' % (str(access_key)),
            'AWS_SECRET_ACCESS_KEY=%s' % (str(secret_key))
        ]
    except Exception as e:
        print('Could not get AWS creds: %s' % e)

    return []


def create_docker_compose(command, tmpdir, hosts, image, additional_volumes, additional_env_vars, customer_script,
                          source_dir, entrypoint):
    services = create_docker_services(command, tmpdir, hosts, image, additional_volumes, additional_env_vars,
                                      customer_script, source_dir, entrypoint)
    content = {
        # docker version on ACC hosts only supports compose 2.1 format
        'version': '2.1',
        'services': services
    }

    y = yaml.dump(content, default_flow_style=False)
    return y


def write_resource_config(path, hosts, current_host):
    content = {
        'current_host': current_host,
        'hosts': hosts
    }

    filename = os.path.join(path, current_host, 'input', 'config', 'resourceconfig.json')
    write_json_file(filename, content)


def write_json_file(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def create_host_names(cluster_size):
    return ['{}-{}'.format(CONTAINER_PREFIX, i) for i in range(1, cluster_size + 1)]


def _read_credentials():
    creds = {}
    credential_file = os.path.expanduser('~/.aws/credentials')
    with open(credential_file) as f:
        for line in f.readlines():
            if not line.startswith('['):
                k, v = line.split('=')
                creds[k.strip().upper()] = v.strip()
    return creds


def check_call(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    _print_cmd(cmd)
    subprocess.check_call(cmd, *popenargs, **kwargs)


def _print_cmd(cmd):
    print('executing docker command: {}{}{}'.format(CYAN_COLOR, ' '.join(cmd), END_COLOR))
    sys.stdout.flush()


def upload_source_files(script, credentials, path=None, job_name='test_job'):
    session = _boto_session(credentials)
    bucket = default_bucket(session)
    s3_source_archive = tar_and_upload_dir(
        session,
        bucket,
        job_name,
        script,
        path)
    return s3_source_archive


def _boto_session(credentials):
    return boto3.Session(aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])


def default_bucket(boto_session):
    """Return the name of the default bucket to use for SageMaker interactions, creating it if necessary.

    Returns:
        str: The name of the default bucket, which will be in the form:
        sagemaker-{AWS account ID}
    """
    s3 = boto_session.resource('s3')
    account = boto_session.client('sts').get_caller_identity()['Account']
    region = boto_session.region_name
    bucket = 'sagemaker-{}-{}'.format(region, account)

    if not bucket_exists(boto_session, bucket):
        try:
            # 'us-east-1' cannot be specified because it is the default region:
            # https://github.com/boto/boto3/issues/125
            if region == 'us-east-1':
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': region})

            print('Created S3 bucket: {}'.format(bucket))
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code != 'BucketAlreadyOwnedByYou':
                raise

    return bucket


def tar_and_upload_dir(session, bucket, job_name, script, directory):
    if directory:
        if not os.path.exists(directory):
            raise ValueError('"{}" does not exist.'.format(directory))
        if not os.path.isdir(directory):
            raise ValueError('"{}" is not a directory.'.format(directory))
        if script not in os.listdir(directory):
            raise ValueError('No file named "{}" was found in directory "{}".'.format(script, directory))
        source_files = [os.path.join(directory, name) for name in os.listdir(directory)]
    else:
        # If no directory is specified, the script parameter needs to be a valid relative path.
        os.path.exists(script)
        source_files = [script]

    print('source files: {}'.format(source_files))
    s3 = session.resource('s3')
    key = '{}/{}'.format(job_name, 'sourcedir.tar.gz')

    with tempfile.TemporaryFile() as f:
        with tarfile.open(mode='w:gz', fileobj=f) as t:
            for sf in source_files:
                # Add all files from the directory into the root of the directory structure of the tar
                t.add(sf, arcname=os.path.basename(sf))
        # Need to reset the file descriptor position after writing to prepare for read
        f.seek(0)
        s3.Object(bucket, key).put(Body=f)

    return 's3://{}/{}'.format(bucket, key)


def bucket_exists(boto_session, bucket_name):
    exists = True
    try:
        s3 = boto_session.resource('s3')
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            exists = False

    return exists


def copy_resource(resource_path, opt_ml_path, relative_src_path, relative_dst_path=None):
    if not relative_dst_path:
        relative_dst_path = relative_src_path

    shutil.copytree(os.path.join(resource_path, relative_src_path), os.path.join(opt_ml_path, relative_dst_path))


def file_exists(resource_folder, file_name, host='algo-1'):
    return os.path.exists(os.path.join(resource_folder, host, file_name))


def load_model(resource_folder, file_name, host='algo-1', serializer=torch):
    with open(os.path.join(resource_folder, host, file_name), 'r') as f:
        return serializer.load(f)


def get_model_dir(resource_folder, host='algo-1'):
    return os.path.join(resource_folder, host)


def request(data, request_type=JSON_CONTENT_TYPE):
    serializer = json if request_type == JSON_CONTENT_TYPE else pickle
    serialized_output = requests.post(REQUEST_URL,
                                      data=serializer.dumps(data),
                                      headers={'Content-type': request_type,
                                               'Accept': request_type}).content

    return serializer.loads(serialized_output)