import numpy as np
import matplotlib.pyplot as plt
from src.dataset import prepare_data
from tensorflow.keras.models import load_model

def predict_diplay_test_set(model, pred_path, num_img):
    _, _, test_images, test_labels = prepare_data()

    pred = model.predict(test_images)

    # Display some test image with pred label
    plt.figure(figsize=(15, 10))
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        pred_label = np.argmax(pred[i])
        true_label = np.argmax(test_labels[i])
        color = 'green' if pred_label == true_label else 'red'
        plt.xlabel(f"Pred: {pred_label}, True: {true_label}", color=color)
    plt.tight_layout()
    plt.savefig(pred_path)
    plt.close()
    print(f"Saved some prediction to {pred_path}")

def main():
    model = load_model('checkpoints/mnist_lenet5.h5')

    predict_diplay_test_set(model, 'output/prediction.png', num_img=100)

if __name__=="__main__":
    main()