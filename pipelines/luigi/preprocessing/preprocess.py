import numpy as np
import click
from io import BytesIO
from urllib.parse import urlparse
import boto3
import os
from pathlib import Path

client = boto3.client("s3")


def normalize_data(image_data: np.array) -> np.array:
    return image_data.astype('float32') / 255.0

# https://stackoverflow.com/questions/48049557/how-to-write-npy-file-to-s3-directly
def from_s3_npy(s3_uri: str):
    bytes_ = BytesIO()
    parsed_s3 = urlparse(s3_uri)
    client.download_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    bytes_.seek(0)
    return np.load(bytes_, allow_pickle=True)

def to_s3_npy(data: np.array, s3_uri: str):
    bytes_ = BytesIO()
    np.save(bytes_, data, allow_pickle=True)
    bytes_.seek(0)
    parsed_s3 = urlparse(s3_uri)
    client.upload_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    return True

@click.command()
@click.argument('s3-uri')
@click.argument('s3-pipeline-folder')
@click.argument('input-file')
@click.argument('output-file')
@click.argument('flag')
def execute_cli(s3_uri, s3_pipeline_folder, input_file, output_file, flag):
    print(f'Input URI: {s3_uri}')
    print(f'Input file: {input_file}')
    data = from_s3_npy(os.path.join(s3_uri, input_file))
    print('Data read!. Processing...')
    data = normalize_data(data)
    print('Data processed!. Uploading...')
    to_s3_npy(data, os.path.join(s3_uri, output_file))
    print('Data uploaded!')

    # marked as SUCCESS for luigi. flag = luigi.Parameter('.SUCCESS_Preprocess')
    # Will write a file called '.SUCCESS_Preprocess' at 's3://bucket/.../')
    # Once done, the task will be flagged as complete the next time it is run in the pipeline
    parsed_s3 = urlparse(str(os.path.join(s3_pipeline_folder, flag)))
    client.put_object(Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:])

if __name__ == "__main__":
    execute_cli()
