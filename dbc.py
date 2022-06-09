from random import shuffle
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Dataset paths
TRAINING_DIR = r"D:/Dog Breed Classification/datasets/training"
VALIDATION_DIR = r"D:/Dog Breed Classification/datasets/testing"
MODEL_DIR = r"D:/Dog Breed Classification/model/model.h5"
TEST_DIR = r"D:/Dog Breed Classification/prediction_img"
# Image properties
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 9

# Generate datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAINING_DIR,
        validation_split=0.1,
        subset="training",
        seed=31,
        image_size=(IMG_HEIGHT,IMG_WIDTH),
        batch_size=9
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DIR,
    validation_split=0.1,
    subset="validation",
    seed=31,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=9
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DIR,
    shuffle=True,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=100
    )

# Randomly flips, rotates, and zooms the training datasets
data_augmentation = keras.Sequential([
            layers.RandomFlip(
                "horizontal",
                input_shape=(IMG_HEIGHT,IMG_WIDTH,3)
                ),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
        ]
        )

class_names = train_ds.class_names
num_of_classes = len(class_names)

# Main loop
while True:
    main_opt = input("Train or Predict? (T/P): ")
    # Predictions
    if main_opt.lower() == "p":
        model = keras.models.load_model(MODEL_DIR)
        model.summary()

        plt.figure(figsize=(20, 20))
        for images, labels in test_ds.take(1):
            for i in range(100):
                predictions = model.predict(images)
                predictions = tf.argmax(predictions, axis=-1)
                ax = plt.subplot(10, 10, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                if class_names[labels[i]] == class_names[predictions[i]]:
                    color = 'g'
                else:
                    color = 'r'
                plt.title(class_names[predictions[i]], color=color)
                plt.axis("off")
        plt.show()
        break
    elif main_opt.lower() == "t":
        # Training
        train_opt = input("Train new model or continue? (N/C): ")
        if train_opt.lower() == "c":
            model = keras.models.load_model(MODEL_DIR)
        elif train_opt.lower() == "n":
            model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_of_classes, activation='softmax')
            ])
        # plt.figure(figsize=(10, 10))
        # for images, labels in train_ds.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        model.summary()

        epochs=50
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        # Showing training results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        save_opt = input("Save model? (Y/N): ")
        if save_opt == "y":
            model.save(MODEL_DIR)
            print("Model Saved!")
            break
        print("Exiting...")
        break
    else:
        print("Please enter a valid option (T/P): ")
