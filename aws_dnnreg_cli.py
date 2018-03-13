from sagemaker.tensorflow import TensorFlowPredictor

print(' ## Building predictor ')
tensor_name = 'TensorName'
predictor = TensorFlowPredictor(tensor_name)

print(' ### Launching predictor ')
features_data=[34.49726772511229, 12.65565114916675, 39.57766801952616, 4.0826206329529615]
result = predictor.predict(features_data)

print(' ### Result ')
print(result)