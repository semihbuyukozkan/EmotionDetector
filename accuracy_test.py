import tensorflow as tf
import tensorflow_hub as hub

data_dir = 'C:\TensorflowModels\dataset_test'
def load_test_data(data_dir, image_size=(224, 224)):
    """
    Load test data from the given directory.
    Assumes data is organized in subdirectories representing class labels.
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    test_data = datagen.flow_from_directory(data_dir,
                                            target_size=image_size,
                                            batch_size=32,
                                            class_mode='sparse')
    return test_data


def evaluate_model(model_path, test_data):
    """
    Load the pretrained model and evaluate it on the test data.
    """
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    loss, accuracy = model.evaluate(test_data)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')


if __name__ == "__main__":
    test_data_dir = 'C:\TensorflowModels\dataset_test'  # Update this with the path to your test dataset
    model_path = 'C:\\TensorflowModels\\emotion_detection_model.h5'

    test_data = load_test_data(test_data_dir)
    evaluate_model(model_path, test_data)