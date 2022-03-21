import luigi
import os
from pathlib import Path
from luigi.contrib.s3 import S3Target

from util import DockerTask

BUCKET = 's3://digits-recognizer-project/'

VERSION = os.getenv('PIPELINE_VERSION', '1.0')

class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'digits-luigi-pipeline/preprocess:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class Preprocess(DockerTask):
    """Task to preprocess images"""

    input_folder = 'input_data/mnist_data/'
    pipeline_folder = 'pipeline/'

    s3_uri = luigi.Parameter(default=os.path.join(BUCKET, input_folder))
    s3_pipeline_folder = luigi.Parameter(default=os.path.join(BUCKET, pipeline_folder))
    input_file = luigi.Parameter(default='samples_1k_X.npy')
    output_file = luigi.Parameter(default='pipeline_processed_samples_1k_X.npy')
    flag = luigi.Parameter('.SUCCESS_Preprocess')

    @property
    def image(self):
        return f'digits-luigi-pipeline/preprocess:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'preprocess.py',
            self.s3_uri,
            self.s3_pipeline_folder,
            self.input_file,
            self.output_file,
            self.flag
        ]

    @property
    def configuration(self):
        task_env_dict = super().configuration['environment']
        task_env_dict.update({'AWS_ACCESS_KEY_ID':os.environ['AWS_ACCESS_KEY_ID']})
        task_env_dict.update({'AWS_SECRET_ACCESS_KEY':os.environ['AWS_SECRET_ACCESS_KEY']})
        task_env_dict.update({'AWS_DEFAULT_REGION':os.environ['AWS_DEFAULT_REGION']})
        print('updated configuration')
        print(super().configuration)
        return super().configuration

    def output(self):
        # Will check if Path s3://.../pipeline/.SUCCESS_Preprocess exists or not, to decide if execute the task
        return S3Target(
            path=os.path.join(self.s3_pipeline_folder, self.flag)
        )


class TrainModel(DockerTask):
    """Task to train a model and register it"""

    input_folder = 'input_data/mnist_data/'
    pipeline_folder = 'pipeline/'

    s3_input_folder = luigi.Parameter(default=os.path.join(BUCKET, input_folder))
    s3_pipeline_folder = luigi.Parameter(default=os.path.join(BUCKET, pipeline_folder))
    input_sample = luigi.Parameter(default='pipeline_processed_samples_1k_X.npy')
    input_labels = luigi.Parameter(default='samples_1k_y.npy')
    mlflow_server_uri = luigi.Parameter(default='http://host.docker.internal:5000')
    flag = luigi.Parameter('.SUCCESS_Training')

    def requires(self):
        return Preprocess()

    @property
    def image(self):
        return f'digits-luigi-pipeline/training:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'train_model.py',
            self.s3_input_folder,
            self.s3_pipeline_folder,
            self.input_sample,
            self.input_labels,
            self.mlflow_server_uri,
            self.flag
        ]

    @property
    def configuration(self):
        task_env_dict = super().configuration['environment']
        task_env_dict.update({'AWS_ACCESS_KEY_ID':os.environ['AWS_ACCESS_KEY_ID']})
        task_env_dict.update({'AWS_SECRET_ACCESS_KEY':os.environ['AWS_SECRET_ACCESS_KEY']})
        task_env_dict.update({'AWS_DEFAULT_REGION':os.environ['AWS_DEFAULT_REGION']})
        return super().configuration

    def output(self):
        # Will check if Path s3://.../pipeline/.SUCCESS_Training exists or not, to decide if execute the task
        return S3Target(
            path=os.path.join(self.s3_pipeline_folder, self.flag)
        )