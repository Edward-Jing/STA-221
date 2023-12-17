import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder, mask=False, target_size=(128, 128)):
    images = []
    if mask:
        for filename in os.listdir(folder):
            if (filename.endswith(".jpg") or filename.endswith(".png")) and "mask" in filename:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, target_size) / 255.0  # 归一化
                images.append(img)
    else:
        for filename in os.listdir(folder):
            if (filename.endswith(".jpg") or filename.endswith(".png")) and "mask" not in filename:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, target_size) / 255.0  # 归一化
                images.append(img)
    return images

def extract_features(images):
    return [img.flatten() for img in images]
