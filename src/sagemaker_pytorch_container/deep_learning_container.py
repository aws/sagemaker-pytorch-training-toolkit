import json
import logging
import requests


def _retrieve_instance_id():
    """
    Retrieve instance ID from instance metadata service
    """
    url = "https://169.254.169.254/latest/meta-data/instance-id"
    response = requests_helper(url, timeout=0.004)
    print(response.text)
    return response.text


def _retrieve_instance_region():
    """
    Retrieve instance region from instance metadata service
    """
    region = None
    url = "https://169.254.169.254/latest/dynamic/instance-identity/document"
    response = requests_helper(url, timeout=0.003)
    if response is not None:
        response_json = json.loads(response.text)
        region = response_json['region']
    return region


def query_bucket():
    """
    GET request on an empty object from an Amazon S3 bucket
    """
    response = None
    instance_id = _retrieve_instance_id()
    region = _retrieve_instance_region()

    if region is not None:
        url = "https://aws-deep-learning-containers-{0}.s3.{0}.amazonaws.com/dlc-containers.txt?x-instance-id={1}".format(region, instance_id)
        response = requests_helper(url, timeout=0.04)

    logging.debug("Tracking finished: {}".format(response))
    return response


def requests_helper(url, timeout):
    response = None
    try:
        response = requests.get(url,timeout=timeout)
    except requests.exceptions.RequestException as e:
        logging.error("Request exception: {}".format(e))

    return response


def main():
    """
    Invoke tracking
    """

    logging.basicConfig(level=logging.ERROR)
    query_bucket()


if __name__ == '__main__':
    main()
