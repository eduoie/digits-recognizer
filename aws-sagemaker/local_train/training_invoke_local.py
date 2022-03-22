# From: https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/tensorflow_script_mode_debug_local_training/tensorflow_script_mode_debug_local_training.py
# This implementation is intented to run on the *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#      `pip install -r requirements.txt`
#   2. Docker Desktop installed and running on your computer:
#      `docker ps`
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
###############################################################################################

from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlow

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    estimator = TensorFlow(entry_point='train.py',
                           source_dir='./',
                           role=DUMMY_IAM_ROLE,
                           instance_count=1,
                           instance_type='local',
                           framework_version='2.7',
                           py_version='py38',
                           )

    training_dataset_path = "file://../../data/initial_data"

    estimator.fit(training_dataset_path)


if __name__ == '__main__':
    main()
