import json
import logging
import requests


def retrieve_instance_metadata():
    """
    Retrieve instance ID from instance metadata service
    """
    url = "http://169.254.169.254/latest/meta-data/instance-id"
    response = requests_helper(url, timeout=0.004)
    return response.text


def retrieve_instance_region():
    """
    Retrieve instance region from instance metadata service
    """
    region = None
    url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
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
    instance_id = retrieve_instance_metadata()
    region = retrieve_instance_region()
    if region is not None:
        url = "https://aws-deep-learning-containers-{0}.s3.{0}.amazonaws.com/dlc-containers.txt?x-instance-id={1}".format(region, instance_id)
        response = requests_helper(url, timeout=0.04)
    logging.debug("Tracking finished: {}".format(response))


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
