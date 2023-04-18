
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

train_DT = pd.read_csv("Train_DT.csv")
train_KNN = pd.read_csv("Train_KNN.csv")
train_LR = pd.read_csv("Train_LR.csv")
train_MLP = pd.read_csv("Train_MLP.csv")
train_NB = pd.read_csv("Train_NB.csv")
train_RF = pd.read_csv("Train_RF.csv")
train_SVC = pd.read_csv("Train_SVC.csv")
train_raw = pd.read_csv("Train.csv")

val_DT = pd.read_csv("Validate_DT.csv")
val_KNN = pd.read_csv("Validate_KNN.csv")
val_LR = pd.read_csv("Validate_LR.csv")
val_MLP = pd.read_csv("Validate_MLP.csv")
val_NB = pd.read_csv("Validate_NB.csv")
val_RF = pd.read_csv("Validate_RF.csv")
val_SVC = pd.read_csv("Validate_SVC.csv")
val_raw = pd.read_csv("Validate.csv")

df_train = pd.DataFrame()
df_train["DT"] = train_DT["Pred"]
df_train["KNN"] = train_KNN["Pred"]
df_train["LR"] = train_LR["Pred"]
df_train["MLP"] = train_MLP["Pred"]
df_train["NB"] = train_NB["Pred"]
df_train["RF"] = train_RF["Pred"]
df_train["SVC"] = train_SVC["Pred"]
# df_train["Class"] = train_raw["Class(Target)"]
# print(df_train)

df_val = pd.DataFrame()
df_val["DT"] = val_DT["Pred"]
df_val["KNN"] = val_KNN["Pred"]
df_val["LR"] = val_LR["Pred"]
df_val["MLP"] = val_MLP["Pred"]
df_val["NB"] = val_NB["Pred"]
df_val["RF"] = val_RF["Pred"]
df_val["SVC"] = val_SVC["Pred"]
# df_val["Class"] = val_raw["Class(Target)"]
# print(df_val)

df_train.replace(to_replace='A', value=0, inplace=True)
df_train.replace(to_replace='B', value=1, inplace=True)
df_train.replace(to_replace='C', value=2, inplace=True)
df_train.replace(to_replace='D', value=3, inplace=True)
# print(df_train)

df_val.replace(to_replace='A', value=0, inplace=True)
df_val.replace(to_replace='B', value=1, inplace=True)
df_val.replace(to_replace='C', value=2, inplace=True)
df_val.replace(to_replace='D', value=3, inplace=True)
# print(df_val)

df_train["Class"] = train_raw["Class(Target)"]
df_val["Class"] = val_raw["Class(Target)"]

# 读取训练数据和标签
X_train = df_train.drop(columns=["Class"]).values
y_train = df_train["Class"].values

# 读取测试数据和标签
X_val = df_val.drop(columns=["Class"]).values
y_val = df_val["Class"].values

"""
0.4153414405986904
"""
model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm='SAMME')

# 训练AdaBoost分类器
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# 输出预测结果和模型评估指标
print("Predictions: ", y_pred)
print("Accuracy: ", accuracy_score(y_val, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_pred))
print("Classification Report: \n", classification_report(y_val, y_pred))

y_pred = model.predict(X_train)

# 输出预测结果和模型评估指标
print("Predictions: ", y_pred)
print("Accuracy: ", accuracy_score(y_train, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_pred))
print("Classification Report: \n", classification_report(y_train, y_pred))


test_DT = pd.read_csv("Test_DT.csv")
test_KNN = pd.read_csv("Test_KNN.csv")
test_LR = pd.read_csv("Test_LR.csv")
test_MLP = pd.read_csv("Test_MLP.csv")
test_NB = pd.read_csv("Test_NB.csv")
test_RF = pd.read_csv("Test_RF.csv")
test_SVC = pd.read_csv("Test_SVC.csv")
test_raw = pd.read_csv("Test.csv")

df_test = pd.DataFrame()
df_test["DT"] = test_DT["Class(Target)"]
df_test["KNN"] = test_KNN["Class(Target)"]
df_test["LR"] = test_LR["Class(Target)"]
df_test["MLP"] = test_MLP["Class(Target)"]
df_test["NB"] = test_NB["Class(Target)"]
df_test["RF"] = test_RF["Class(Target)"]
df_test["SVC"] = test_SVC["Class(Target)"]

df_test.replace(to_replace='A', value=0, inplace=True)
df_test.replace(to_replace='B', value=1, inplace=True)
df_test.replace(to_replace='C', value=2, inplace=True)
df_test.replace(to_replace='D', value=3, inplace=True)

pred = model.predict(df_test)
test_raw['Class(Target)'] = pred
test_raw.to_csv("Test_Integration_Ada.csv", index=False)