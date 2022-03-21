export AWS_ACCESS_KEY_ID := $(shell cat ${HOME}/.aws/credentials | grep aws_access_key_id | cut -f2 -d"=" | cut -f2 -d" ")
export AWS_SECRET_ACCESS_KEY := $(shell cat ${HOME}/.aws/credentials | grep aws_secret_access_key | cut -f2 -d"=" | cut -f2 -d" ")
export AWS_DEFAULT_REGION=$(shell cat ${HOME}/.aws/config | grep region | cut -f2 -d"=" | cut -f2 -d" ")

.PHONY: help

help: ## Print available commands.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

prepare_conda_environment:
	conda env create -f ../environment.yml && \
	conda activate digits-recognizer

build_preprocess:
	docker build -t digits-preprocess:1.0 preprocessing

debug_preprocess: ## opens the container for debugging
	docker run \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
     	-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
        -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
		--rm -ti --name digits-preprocessing digits-preprocess:1.0 /bin/bash

run_preprocess:
	docker run \
        -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
        -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
        -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
        --rm --name digits-preprocessing digits-preprocess:1.0 \
	    python /src/preprocess.py s3://digits-recognizer-project/input_data/mnist_data/ samples_1k_X.npy processed_samples_1k_X.npy

build_training:
	docker build -t digits-train_model:1.0 training

debug_training: ## opens the container for debugging. Adds a volume so changes in code files persists
	docker run \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
     	-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
        -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
		-v ${PWD}/training:/training \
		--network=docker_mlnetwork \
		--rm -ti --name digits-training-model digits-train_model:1.0 /bin/bash

run_training:
	docker run \
        -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
        -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
        -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
        --rm --name digits-training-model digits-train_model:1.0 \
	    python /src/train_model.py s3://digits-recognizer-project/input_data/mnist_data/ \
		processed_samples_1k_X.npy samples_1k_y.npy http://host.docker.internal:5000
