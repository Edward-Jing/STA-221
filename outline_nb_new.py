import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

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
base_path = '/Users/kirito/Desktop/STA 221/Group Project/codes/Dataset_BUSI_with_GT'
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

# # 高斯先验分布的朴素贝叶斯模型
# gnb = GaussianNB(var_smoothing=4e-3)
# gnb.fit(X_train, y_train)

# # 模型评估
# y_pred = gnb.predict(X_test)
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=categories))

# # 训练伯努利先验分布的朴素贝叶斯模型（结果没有高斯先验的朴素贝叶斯好）
# bnb = BernoulliNB(alpha=1e-8)
# bnb.fit(X_train, y_train)

# # 模型评估
# y_pred = bnb.predict(X_test)
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=categories))

# 设置精细的朴素贝叶斯模型网格
param_grid = {
    'var_smoothing': np.logspace(-1, -5, num=100)
}

# 创建高斯先验分布的朴素贝叶斯模型
gnb = GaussianNB()

# 使用网格搜索
grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("\nBest Parameters:", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_gnb = grid_search.best_estimator_
y_pred = best_gnb.predict(X_test)

# 模型评估
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# # 设置精细的朴素贝叶斯模型网格
# param_grid = {
#     'alpha': np.logspace(-8, -10, num=50)
# }

# # 创建伯努利先验分布的朴素贝叶斯模型
# bnb = BernoulliNB()

# # 使用网格搜索
# grid_search = GridSearchCV(estimator=bnb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # 输出最佳参数
# print("\nBest Parameters:", grid_search.best_params_)

# # 使用最佳参数的模型进行预测
# best_gnb = grid_search.best_estimator_
# y_pred = best_gnb.predict(X_test)

# # 模型评估
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))