import os
import tensorflow as tf
import numpy as np
import mlflow
import click
import boto3

from io import BytesIO
from urllib.parse import urlparse

client = boto3.client("s3")

def from_s3_npy(s3_uri: str):
    bytes_ = BytesIO()
    parsed_s3 = urlparse(s3_uri)
    client.download_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    bytes_.seek(0)
    return np.load(bytes_, allow_pickle=True)


def train_model(train_X, train_y, valid_X, valid_y):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_X,
        train_y,
        epochs=10,
        batch_size=128,
        validation_data=(valid_X, valid_y),
        verbose=1,
    )


@click.command()
@click.argument('input-dir')
@click.argument('s3-pipeline-folder')
@click.argument('input-sample')
@click.argument('input-labels')
@click.argument('mlflow-server-uri')
@click.argument('flag')
def execute_cli(input_dir, s3_pipeline_folder, input_sample, input_labels, mlflow_server_uri, flag):

    print("INPUTS\n=====")
    print(input_dir)
    print(input_sample)
    print(input_labels)

    SAMPLES = os.path.join(input_dir, input_sample)
    SAMPLES_LABELS = os.path.join(input_dir, input_labels)

    train_data = from_s3_npy(SAMPLES)
    label_data = from_s3_npy(SAMPLES_LABELS)

    split_point = int(len(train_data) * 0.9)
    train_X, train_y = train_data[:split_point], label_data[:split_point]
    valid_X, valid_y = train_data[split_point:], label_data[split_point:]

    print('Data read!')
    print(train_X.shape, valid_X.shape)

    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_experiment("simple-keras-docker-2")
    mlflow.tensorflow.autolog()
    
    print('Training model')
    with mlflow.start_run():
        mlflow.log_param('sample_size', len(train_data))
        mlflow.log_param('train_valid_split_point', split_point)
        # mlflow.log_artifact(os.path.join(output_dir, input_sample))
        # mlflow.log_artifact(os.path.join(output_dir, input_labels))
        train_model(train_X, train_y, valid_X, valid_y) 
    print('Model trained')
    mlflow.tensorflow.autolog(disable=True)

    # obtain runs with specific metrics
    runs_df = mlflow.search_runs(filter_string="metrics.val_accuracy > 0.5")
    # obtain best run
    run_id = runs_df.loc[runs_df['metrics.val_accuracy'].idxmax()]['run_id']
    mlflow.register_model(
        f"runs:/{run_id}",
        "best-keras-digits-model"
    )

    # marked as SUCCESS for luigi. flag = luigi.Parameter('.SUCCESS_Training')
    parsed_s3 = urlparse(str(os.path.join(s3_pipeline_folder, flag)))
    client.put_object(Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:])


if __name__ == "__main__":
    # Do not name it as cli(), it is used internally by debugpy in VSCode
    execute_cli()
