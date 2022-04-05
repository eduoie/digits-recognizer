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
import numpy as np

from PIL import Image, ImageOps

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def do_inference_on_local_endpoint(predictor):
    print(f'\nStarting Inference on endpoint (local).')

    x_test = np.load('../../data/initial_data/samples_1k_X.npy')
    y_test = np.load('../../data/initial_data/samples_1k_y.npy')

    # apply preprocessing
    x_test = x_test.astype('float32') / 255.0

    results = predictor.predict(x_test[:10])['predictions']
    # flat_list = [float('%.1f' % (item)) for sublist in results for item in sublist]
    # print('predictions: \t{}'.format(np.array(flat_list)))
    # print('target values: \t{}'.format(y_test[:10].round(decimals=1)))

    # predict an unseen image
    im2 = Image.open("2_87.jpg").convert('L')
    im2 = ImageOps.invert(im2)
    thresh = 150
    fn = lambda x: x if x > thresh else 0
    im2 = im2.point(fn, mode='L')
    im2 = im2.resize((28,28))
    pic = np.array(im2)
    print(pic.shape)
    pic = pic.astype('float32') / 255.0

    results = predictor.predict(pic)
    print(results)



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

    print('Deploying endpoint in local mode')
    predictor = estimator.deploy(initial_instance_count=1, instance_type='local')

    do_inference_on_local_endpoint(predictor)

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)



if __name__ == '__main__':
    main()
