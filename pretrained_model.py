import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import tensorflow_hub as hub
def model_creator():
    model_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

    pre_trained_model = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
        pre_trained_model,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(7, activation='softmax')
    ])


    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )

    model.summary()

    model.save('C:\TensorflowModels\emotion_detection_model.h5', save_format='h5')

if __name__ == "__main__":
    model_creator()