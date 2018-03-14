import tensorflow as tf


def predict(predict_fn, features_data):
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
    predictions = predict_fn({"inputs":[input]})
    print(predictions['outputs'])


def main():
    print(' # Loading ')
    export_dir = 'SavedModelDir'
    predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

    print(' # Prediction #1 ')
    features_data=[34.49726772511229, 12.65565114916675, 39.57766801952616, 4.0826206329529615]
    predict(predict_fn, features_data)

    print(' # Prediction #2 ')
    features_data=[31.9262720264, 11.1094607287, 37.2689588683, 2.6640341821]
    predict(predict_fn, features_data)

    print(' # Prediction #3 ')
    features_data=[33.0009147556, 11.3302780578, 37.1105974421, 4.1045432024]
    predict(predict_fn, features_data)

    print(' # Prediction #3 ')
    features_data=[34.3121669974, 11.8105867646, 37.4141335747, 2.4735961208]
    predict(predict_fn, features_data)

if __name__ == "__main__":
    main()
