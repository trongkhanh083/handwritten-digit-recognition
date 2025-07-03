import tensorflow as tf
from src.model import lenet5_model
from src.dataset import prepare_data
import os
import pickle

def train():
    train_images, train_labels, _, _ = prepare_data()

    model = lenet5_model()

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2)

    os.makedirs('output', exist_ok=True)
    history_path = 'output/history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"Saved training history to {history_path}")

    return history

def main():
    train()

if __name__=="__main__":
    main()