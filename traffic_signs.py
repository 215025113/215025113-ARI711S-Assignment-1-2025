import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Constants
IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CLASSES = 43


def load_data(data_dir):
    print("Loading data...")
    images = []
    labels = []

    for label in range(NUM_CLASSES):
        label_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(label_dir):
            continue

        for filename in os.listdir(label_dir):
            if filename.endswith(".ppm") or filename.endswith(".png") or filename.endswith(".jpg"):
                try:
                    img_path = os.path.join(label_dir, filename)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    print("Data loaded.")
    return np.array(images), np.array(labels)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic_signs.py data_directory [model.h5]")

    data_dir = sys.argv[1]
    model_filename = sys.argv[2] if len(sys.argv) == 3 else None

    # Load and preprocess data
    X, y = load_data(data_dir)
    X = X.astype("float32") / 255.0  # normalize
    y_cat = to_categorical(y, NUM_CLASSES)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # Train model
    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

    # Save model if filename given
    if model_filename:
        model.save(model_filename)
        print(f"Model saved to {model_filename}")

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Model accuracy: {test_acc:.4f}")

    # Confusion matrix
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
