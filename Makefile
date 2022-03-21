# https://stackoverflow.com/questions/1909188/define-make-variable-at-rule-execution-time
export AWS_ACCESS_KEY_ID := $(shell cat ${HOME}/.aws/credentials | grep aws_access_key_id | cut -f2 -d"=" | cut -f2 -d" ")
export AWS_SECRET_ACCESS_KEY := $(shell cat ${HOME}/.aws/credentials | grep aws_secret_access_key | cut -f2 -d"=" | cut -f2 -d" ")
export AWS_DEFAULT_REGION=$(shell cat ${HOME}/.aws/config | grep region | cut -f2 -d"=" | cut -f2 -d" ")

.PHONY: help

help: ## Print available commands.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build_jupyter: ## a jupyter lab with some libraries for distributed working preinstalled
	docker build -t notebooks_digits:1.0 docker/notebooks

run_jupyter: ## run Jupyter lab https://github.com/jupyter/docker-stacks
	docker run -it --rm \
		--network docker_mlnetwork \
		-p 8888:8888/tcp \
		-v "${PWD}/notebooks":/home/jovyan/work \
		-v "${PWD}/data":/home/jovyan/data \
		-v ${PWD}/src:/home/jovyan/src \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
     	-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
        -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
		--name notebooks_container \
		notebooks_digits:1.0

run_mlflow: ## run the MLFlow server
	docker compose -f docker/docker-compose.yml up --build -d

stop_mlflow:
	docker compose -f docker/docker-compose.yml down
