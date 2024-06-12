import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


#model_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#base_model.summary()

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(7, activation='softmax')(x)

model = Model(inputs= base_model.input, outputs = x)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )


model.summary()