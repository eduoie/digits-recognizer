export AWS_ACCESS_KEY_ID := $(shell cat ${HOME}/.aws/credentials | grep aws_access_key_id | cut -f2 -d"=" | cut -f2 -d" ")
export AWS_SECRET_ACCESS_KEY := $(shell cat ${HOME}/.aws/credentials | grep aws_secret_access_key | cut -f2 -d"=" | cut -f2 -d" ")
export AWS_DEFAULT_REGION=$(shell cat ${HOME}/.aws/config | grep region | cut -f2 -d"=" | cut -f2 -d" ")

build_pipeline_images:
	./build_images.sh 1.0

run_pipeline:
	docker compose -f docker-compose.yml up --build -d
