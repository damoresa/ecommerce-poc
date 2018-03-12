import boto3
from sagemaker.tensorflow import TensorFlow

region = boto3.Session().region_name
print(' ## Region ' + region)

print(' ### Allocating SageMaker instance ')
tf_estimator = TensorFlow(entry_point='aws-ecommerce-poc-dnn.py', role='SECRET',
                          training_steps=10000, evaluation_steps=100,
                          train_instance_count=1, train_instance_type='ml.c4.8xlarge',
                          output_path='s3://hackathon-pa3mm/ecommerce-poc/output')

print(' ### Fitting model')
tf_estimator.fit(inputs='s3://hackathon-pa3mm/ecommerce-poc/')