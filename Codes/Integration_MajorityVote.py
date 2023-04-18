import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense

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
# df_train["RF1"] = train_RF["Pred"]
# df_train["RF2"] = train_RF["Pred"]
# df_train["RF3"] = train_RF["Pred"]
# df_train["RF4"] = train_RF["Pred"]
# df_train["RF5"] = train_RF["Pred"]
# df_train["RF6"] = train_RF["Pred"]
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
# df_val["RF1"] = val_RF["Pred"]
# df_val["RF2"] = val_RF["Pred"]
# df_val["RF3"] = val_RF["Pred"]
# df_val["RF4"] = val_RF["Pred"]
# df_val["RF5"] = val_RF["Pred"]
# df_val["RF6"] = val_RF["Pred"]
df_val["SVC"] = val_SVC["Pred"]
# df_val["Class"] = val_raw["Class(Target)"]
# print(df_val)

df = pd.DataFrame()
df['majority_vote'] = df_train.apply(lambda x: x.mode()[0], axis=1)
df['Class'] = train_raw["Class(Target)"]
# print(df)
accuracy = (df.iloc[:,0] == df.iloc[:,1]).sum() / len(df)
print(accuracy)

df_s = pd.DataFrame()
df_s['majority_vote'] = df_val.apply(lambda x: x.mode()[0], axis=1)
df_s['Class'] = val_raw["Class(Target)"]
# print(df_s)
accuracy = (df_s.iloc[:,0] == df_s.iloc[:,1]).sum() / len(df_s)
print(accuracy)

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

df_t = pd.DataFrame()
df_t['majority_vote'] = df_val.apply(lambda x: x.mode()[0], axis=1)
test_raw["Class(Target)"] = df_t['majority_vote']
test_raw.to_csv("Test_Integration_MajorityVote.csv", index=False)