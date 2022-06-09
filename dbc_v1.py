import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image as im
import glob

class_dict = {
    0: "Doberman",
    1: "German Shepherd",
    2: "Golden Retriever",
    3: "Pug",
    4: "Samoyed",
    5: "Shiba Inu",
    6: "Siberian Husky"
}

TRAINING_DIR = r"D:/Dog Breed Classification/datasets/training"
VALIDATION_DIR = r"D:/Dog Breed Classification/datasets/testing"
MODEL_DIR = r"D:/Dog Breed Classification/model/model.h5"
TEST_DIR = r"D:/Dog Breed Classification/prediction_img/*"

training_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
train_generator, validation_generator = None, None
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode = 'categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size = (150,150),
    class_mode = 'categorical'
)

def start_training(mode):
    if mode == "n":
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
        ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        verbose=1
    )
    model.save(MODEL_DIR)
    model.summary()
    print("Training Complete, model saved!")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
menu = input("Train or Predict? (t/p): ")
if menu == "t":
    mode_of_training = input("New or Continue (n/c): ")
    if mode_of_training == "c": 
        model = tf.keras.models.load_model(MODEL_DIR)
        print("Model loaded!")
    while mode_of_training != "n":
        options = input("[s] for model summary, [any key] for start training: ")
        if options == "s":
            model.summary()
        else:
            break
    start_training(mode_of_training)
else:
    model = tf.keras.models.load_model(MODEL_DIR)
    print("Model loaded!")
    path = TEST_DIR
    image_arr = []
    images = glob.glob(TEST_DIR, recursive=True)
    for image in images:
        img = im.open(image).resize((150, 150))
        img = np.expand_dims(img, axis=0)
        img = np.array(img)
        image_arr.append(img)
    
    image_arr = np.vstack(image_arr)
    classes = model.predict(image_arr, batch_size=10)
    
    for c in classes:
        print(class_dict[int(np.where(c == 1)[0])])