from PIL import Image as im
import PIL
import glob
import os

TRAINING_DIR = r"D:/Dog Breed Classification/datasets/training/Siberian Husky/*"
VALIDATION_DIR = r"D:/Dog Breed Classification/datasets/testing/Samoyed/*"

images = glob.glob(VALIDATION_DIR, recursive=True)
for image in images:
    try:
        img = im.open(image)
        img.load()
    except PIL.UnidentifiedImageError as e:
        print("Corrupted Image: " + str(e).split("\\")[-1][0:-1])