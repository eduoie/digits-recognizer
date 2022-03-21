#!/usr/bin/env bash
if test -z "$1"
then
      echo "Usage ./build-task-images.sh VERSION"
      echo "No version was passed! Please pass a version to the script e.g. 0.1"
      exit 1
fi

VERSION=$1

docker build -t digits-luigi-pipeline/base-docker base_image
docker build -t digits-luigi-pipeline/preprocess:$VERSION preprocessing
docker build -t digits-luigi-pipeline/training:$VERSION training

docker build -t digits-luigi-pipeline/orchestrator:$VERSION orchestrator