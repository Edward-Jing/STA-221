import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and not filename.endswith("_mask.jpg") and not filename.endswith("_mask.png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            img_normalized = img / 255.0  # 归一化
            images.append(img_normalized)
    return images

def extract_features(images):
    return [img.flatten() for img in images]

# 路径设置
base_path = 'C:/Users/Edward/Desktop/2023 Fall/STA 221/STA 221 Project/Dataset_BUSI_with_GT'
categories = ['benign', 'malignant', 'normal']

# 遍历每个分类
all_features = []
for category in categories:
    folder_path = os.path.join(base_path, category)
    images = load_images_from_folder(folder_path)
    features = extract_features(images)
    all_features.extend(features)

# 转换为NumPy数组
features_matrix = np.array(all_features)
# print(features_matrix.shape)

# 生成标签
labels = []
for category in categories:
    folder_path = os.path.join(base_path, category)
    images = load_images_from_folder(folder_path)
    labels.extend([category] * len(images))

# 将标签转换为数字
label_dict = {'benign': 0, 'malignant': 1, 'normal': 2}
encoded_labels = [label_dict[label] for label in labels]

# 划分训练集和测试集 是一个2/8分
X_train, X_test, y_train, y_test = train_test_split(features_matrix, encoded_labels, test_size=0.2, random_state=42)

# 生成图像特征和标签
all_features = []
all_labels = []
for category in categories:
    folder_path = os.path.join(base_path, category)
    images = load_images_from_folder(folder_path)
    features = extract_features(images)
    all_features.extend(features)
    all_labels.extend([category] * len(images))

# 将标签转换为数字
label_dict = {'benign': 0, 'malignant': 1, 'normal': 2}
encoded_labels = [label_dict[label] for label in all_labels]

# 转换为NumPy数组
features_matrix = np.array(all_features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_matrix, encoded_labels, test_size=0.2, random_state=42)

# 训练随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型评估
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))