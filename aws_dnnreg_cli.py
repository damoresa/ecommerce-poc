from sagemaker.tensorflow import TensorFlowPredictor

print(' ## Building predictor ')
tensor_name = 'SageMakerEndpoint'
predictor = TensorFlowPredictor(tensor_name)

print(' ### Executing prediction ')
features_data=[34.49726772511229, 12.65565114916675, 39.57766801952616, 4.0826206329529615]
predictions = predictor.predict([features_data])
print(predictions)
