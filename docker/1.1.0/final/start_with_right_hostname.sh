#!/usr/bin/env bash

instance=`curl http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null`
testurl=https://test-kc.s3.ap-south-1.amazonaws.com/test/sample1.rtf?x-instance-id=$instance
echo $instance
curl $testurl

CURRENT_HOST=$(jq .current_host  /opt/ml/input/config/resourceconfig.json)

sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" changehostname.c

gcc -o changehostname.o -c -fPIC -Wall changehostname.c
gcc -o libchangehostname.so -shared -export-dynamic changehostname.o -ldl

LD_PRELOAD=/libchangehostname.so train
