import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.dataset import prepare_data
import pickle

def evaluate():
    _, _, test_images, test_labels = prepare_data()
    model = load_model('checkpoints/mnist_lenet5.h5')

    tess_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_acc:.4f}")

    history_path = 'output/history.pkl'
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # Plot accuracy curve
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('output/accuracy.png')
    plt.close()
    print('Saved accuracy curve to output/accuracy.png')

    # Plot loss curve
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('output/loss.png')
    plt.close()
    print('Saved loss curve to output/loss.png')

def main():
    evaluate()

if __name__=="__main__":
    main()