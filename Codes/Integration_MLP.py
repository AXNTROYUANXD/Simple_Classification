import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Define the MLP model
model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on your training data
X_train = df_train.drop(columns=['Class'])
y_train = pd.get_dummies(df_train['Class'])
model.fit(X_train, y_train, epochs=200, batch_size=128)
model.save("MLP.h5")

# Evaluate the model on your validation data
X_val = df_val.drop(columns=['Class'])
y_val = pd.get_dummies(df_val['Class'])
loss, accuracy = model.evaluate(X_val, y_val)
y_pred = model.predict(X_val)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)


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
# Convert the predicted values to class labels
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
pred_labels = [class_labels[np.argmax(row)] for row in pred]
test_raw['Class(Target)'] = pred_labels
test_raw.to_csv("Test_Integration_MLP.csv", index=False)