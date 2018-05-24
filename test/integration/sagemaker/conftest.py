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
import pytest


@pytest.fixture(autouse=True)
def skip_by_device_type(request, tag):
    if (request.node.get_marker('skip_gpu') and 'gpu' in tag) or (request.node.get_marker('skip_cpu') and 'cpu' in tag):
        pytest.skip('Skipping because tag is: {}'.format(tag))
