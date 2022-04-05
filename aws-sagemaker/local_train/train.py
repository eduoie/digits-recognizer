import tensorflow as tf
import argparse
import os
import numpy as np
import json


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

    return model


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # this is an annoying difference between local and SageMaker training parameters
    if os.environ.get('SM_CHANNEL_TRAINING') is None: # if none, environment is on SageMaker
        print('Environment is SageMaker')
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN')) # sagemaker
    else:
        print('Environment is Local')
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING')) # local

    # parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    # train_data, train_labels = _load_training_data(args.train)
    # eval_data, eval_labels = _load_testing_data(args.train)

    # mnist_classifier = model(train_data, train_labels, eval_data, eval_labels)

    print(args)

    X_data = np.load(os.path.join(args.train, 'samples_1k_X.npy'))
    y_data = np.load(os.path.join(args.train, 'samples_1k_y.npy'))

    X_train = X_data.astype('float32') / 255.0

    split_point = int(len(X_data) * 0.9)
    train_X, train_y = X_data[:split_point], y_data[:split_point]
    valid_X, valid_y = X_data[split_point:], y_data[split_point:]

    model = train_model(train_X, train_y, valid_X, valid_y)
    path = os.path.join(args.sm_model_dir, '/1')
    # save model
    # model.save(args.sm_model_dir + '/1')
    # print(f'Model will be saved to: {path}')


    print(f'Model will be saved to: {args.sm_model_dir + "/1"}')
    model.save(args.sm_model_dir + '/1')

    # https://www.tensorflow.org/guide/saved_model
    # The save-path follows a convention used by TensorFlow Serving where the last path component (1/ here) is a
    # version number for your model - it allows tools like Tensorflow Serving to reason about the relative freshness.
    # model.save(path)
