from sagemaker import RealTimePredictor
from sagemaker.tensorflow.predictor import tf_serializer, tf_deserializer
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow_serving.apis import regression_pb2

import tensorflow as tf

print(' # Preparing data ')
features_data=[34.49726772511229, 12.65565114916675, 39.57766801952616, 4.0826206329529615]
model_input = tf.train.Example(
    features=tf.train.Features(
        feature={
            'inputs': tf.train.Feature(
                float_list=tf.train.FloatList(value=features_data)
            )
        }
    )
)

print(' # Preparing predictor ')
endpoint='SageMakerEndpoint'

predictor=RealTimePredictor(
    endpoint=endpoint,
    deserializer=tf_deserializer,
    serializer=tf_serializer,
    content_type='application/octet-stream')

print(' # Preparing request ')
request=regression_pb2.RegressionRequest()
request.model_spec.name='generic_model'
request.model_spec.signature_name=DEFAULT_SERVING_SIGNATURE_DEF_KEY
request.input.example_list.examples.extend([model_input])

print(' # Invoking SageMaker API ')
result=predictor.predict(request)

print(result)