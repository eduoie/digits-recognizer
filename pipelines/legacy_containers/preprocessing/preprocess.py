import numpy as np
import click
from io import BytesIO
from urllib.parse import urlparse
import boto3
import os

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
@click.argument('input-file')
@click.argument('output-file')
def execute_cli(s3_uri, input_file, output_file):
    print(f'Input URI: {s3_uri}')
    print(f'Input file: {input_file}')
    data = from_s3_npy(os.path.join(s3_uri, input_file))
    print('Data read!')
    data = data / 255.0
    to_s3_npy(data, os.path.join(s3_uri, output_file))
    print('Data processed and uploaded!')

    # with open(input_data, 'rb') as f:
    #     data = np.load(f)
    # data = data / 255.0
    
    # with open(output_path, 'wb') as f:
    #     np.save(f, data)

if __name__ == "__main__":
    execute_cli()
