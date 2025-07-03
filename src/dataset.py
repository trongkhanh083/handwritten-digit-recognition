import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def prepare_data():
    # Load MNIST dataset (60K train + 10K test)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel value to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape image to include channel dimension (28, 28, 1)
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    # Convert label to one-hot encoding
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print(train_images.shape)
    print(test_images.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    return train_images, train_labels, test_images, test_labels