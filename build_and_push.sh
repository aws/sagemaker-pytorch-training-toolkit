#!/usr/bin/env bash

#Build base image for GPU with,
# docker build -t pytorch-base:1.0.0-gpu-py3 -f docker/1.0.0/base/Dockerfile.gpu --build-arg py_version=3 .
#Build base image for CPU with,
# docker build -t pytorch-base:1.0.0-cpu-py3 -f docker/1.0.0/base/Dockerfile.cpu --build-arg py_version=3 .

arg="$1"


account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

if [ "$arg" = "cpu" ]
then
        algorithm_name=sagemaker-pytorch-cpu-py3
        echo ${algorithm_name}
elif [ "$arg" = "gpu" ]
then
        # The name of our algorithm
        algorithm_name=sagemaker-pytorch-gpu-py3-with-apex
        echo ${algorithm_name}
else
        echo cpu or gpu arguments must be passed to the command!
        exit
fi

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the container repository for doesn't exist in ECR, create it.

# Create repository for docker image
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi


# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order
# to detect your network configuration correctly.  (This is a known issue.)
if [ -d "/home/ec2-user/SageMaker" ]; then
  sudo service docker restart
fi

#Creating wheel file from setup.py
python setup.py bdist_wheel


if [ "$arg" = "cpu" ]
then
        # Build CPU Docker image
        docker build -t ${algorithm_name} -f docker/1.0.0/final/Dockerfile.cpu --build-arg py_version=3 .
elif [ "$arg" = "gpu" ]
then
        # Build GPU Docker image
        docker build -t ${algorithm_name} -f docker/1.0.0/final/Dockerfile.gpu --build-arg py_version=3 .
else
        echo cpu or gpu arguments must be passed to the command!
        exit
fi


docker tag ${algorithm_name} ${fullname}

docker push ${fullname}

echo Image Successfully Pushed to ECR

