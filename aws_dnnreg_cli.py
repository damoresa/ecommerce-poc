from sagemaker.tensorflow import TensorFlowPredictor

import tensorflow as tf

print(' ## Building predictor ')
tensor_name = 'EndpointName'
predictor = TensorFlowPredictor(tensor_name)

print(' ### Executing prediction ')
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
input=model_input.SerializeToString()
predictions = predictor.predict({'inputs':[features_data]})
print(predictions['outputs'])
