# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import unittest
import requests

import requests_mock
import pytest


from sagemaker_pytorch_container import deep_learning_container as deep_learning_container_to_test


@pytest.fixture(name='fixture_instance_id')
def fixture_instance_id(requests_mock):
    return requests_mock.get('https://169.254.169.254/latest/meta-data/instance-id', text = 'i123')


@pytest.fixture(name='fixture_region')
def fixture_region(requests_mock):
    return requests_mock.get('https://169.254.169.254/latest/dynamic/instance-identity/document', json ={'region': 'test'})


@requests_mock.mock()
def test_retrieve_instance_id(requests_mock):
   requests_mock.get('https://169.254.169.254/latest/meta-data/instance-id', text='i123')
   result = deep_learning_container_to_test._retrieve_instance_id()
   assert 'i123' == result


@requests_mock.mock()
def test_retrieve_region(requests_mock):
   requests_mock.get('https://169.254.169.254/latest/dynamic/instance-identity/document', json={'region': 'test'})
   result = deep_learning_container_to_test._retrieve_instance_region()
   assert 'test' == result


def test_query_bucket(requests_mock, fixture_region,fixture_instance_id):
    fixture_instance_id.return_value = 'i123'
    fixture_region.return_value = 'test'
    requests_mock.get('https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123', text = 'Access Denied')
    actual_response = deep_learning_container_to_test.query_bucket()
    assert 'Access Denied' == actual_response.text


def test_query_bucket_region_none(requests_mock, fixture_region,fixture_instance_id):
    fixture_instance_id.return_value = 'i123'
    fixture_region.return_value = None
    requests_mock.get(
        'https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123')
    actual_response = deep_learning_container_to_test.query_bucket()
    assert not actual_response.text


def test_HTTP_error_on_S3(requests_mock, fixture_region,fixture_instance_id):
    fixture_instance_id.return_value = 'i123'
    fixture_region.return_value = 'test'
    requests_mock.get(
        'https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123',
        exc=requests.exceptions.HTTPError)
    requests_mock.side_effect = requests.exceptions.HTTPError

    with pytest.raises(requests.exceptions.HTTPError):
        actual_response = requests.get('https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123')

        assert None == actual_response


def test_connection_error_on_S3(requests_mock, fixture_region,fixture_instance_id):
    fixture_instance_id.return_value = 'i123'
    fixture_region.return_value = 'test'
    requests_mock.get(
        'https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123',
        exc=requests.exceptions.ConnectionError)

    with pytest.raises(requests.exceptions.ConnectionError):
        actual_response = requests.get(
            'https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123')

        assert None == actual_response


def test_timeout_error_on_S3(requests_mock, fixture_region,fixture_instance_id):
    fixture_instance_id.return_value = 'i123'
    fixture_region.return_value = 'test'
    requests_mock.get(
        'https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123',
        exc=requests.Timeout)

    with pytest.raises(requests.exceptions.Timeout):
        actual_response = requests.get(
            'https://aws-deep-learning-containers-test.s3.test.amazonaws.com/dlc-containers.txt?x-instance-id=i123')

        assert None == actual_response


if __name__ == '__main__':
    unittest.main()
