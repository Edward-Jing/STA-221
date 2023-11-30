import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# def load_mask_images_from_folder(folder, target_size=(128, 128)):
#     images = []
#     for filename in os.listdir(folder):
#         if filename.endswith("_mask.jpg") or filename.endswith("_mask.png"):
#             img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, target_size)/ 255.0  # 归一化
#             images.append(img)
#     return images

def load_images_from_folder(folder, mask=False, target_size=(64, 64)):
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

# 路径设置
base_path = 'C:/Users/Edward/Desktop/2023 Fall/STA 221/STA 221 Project/Dataset_BUSI_with_GT'
categories = ['benign', 'malignant', 'normal']

# 生成图像特征和标签
all_features = []
all_labels = []
for category in categories:
    folder_path = os.path.join(base_path, category)
    images = load_images_from_folder(folder_path, mask=True)
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
#
# # 训练随机森林模型
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
#
# # 模型评估
# y_pred = clf.predict(X_test)
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# 设置随机森林参数网格
# 设置精细的随机森林参数网格
param_grid = {
    'n_estimators': [70, 75, 80, 85, 100, 200],
    'max_depth': [9, 10, 11, 12],
    'min_samples_split': [4, 5, 6, 7, 8, 9, 10]
}
# 创建随机森林模型
rf = RandomForestClassifier(random_state=42)

# 使用网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)
# Best Parameters: {'max_depth': 9, 'min_samples_split': 4, 'n_estimators': 70}

# 使用最佳参数的模型进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 模型评估
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))