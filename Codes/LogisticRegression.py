import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

train_df = pd.read_csv("Processed_Train(1).csv")
val_df = pd.read_csv("Processed_Validate(1).csv")
test_df = pd.read_csv("Processed_Test(1).csv")

# plotdata = sns.pairplot(train_df, hue='Class(Target)')
# plotdata.fig.suptitle("Pair Plot Analysis", y=1.08)

x_train = train_df[['Gender', 'Married', 'Age', 'Graduate', 'Years_of_Working ', 'Spending_Score', 'Family_Members', 'Category', 'Profession']].values
y_train = train_df[['Class(Target)']].values

x_val = val_df[['Gender', 'Married', 'Age', 'Graduate', 'Years_of_Working ', 'Spending_Score', 'Family_Members', 'Category', 'Profession']].values
y_val = val_df[['Class(Target)']].values

x_test = test_df[['Gender', 'Married', 'Age', 'Graduate', 'Years_of_Working ', 'Spending_Score', 'Family_Members', 'Category', 'Profession']].values
# y_test = test_df[:, 10].values

model = LogisticRegression(max_iter=600)
model.fit(x_train, y_train)

"""
Accuracy :  0.44059869036482696 (on validation set)
"""
y_pred_lr_val = model.predict(x_val)
print("Accuracy : ", accuracy_score(y_val, y_pred_lr_val))
cr = classification_report(y_val, y_pred_lr_val)
print("\t\tClassification Report\n" + "--"*28 + "\n", cr)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_val, y_pred_lr_val), display_labels=['A', 'B', 'C', 'D'])
cm_display.plot()
plt.show()

y_pred_lr_test = model.predict(x_test)

print(y_pred_lr_test)
test_df['Class(Target)'] = y_pred_lr_test
test_df.to_csv("Test_LR.csv", index=False)

test_raw = pd.read_csv('Test.csv')
test_raw["Class(Target)"] = y_pred_lr_test
test_raw.to_csv('Test_LR.csv', index=False)

val_raw = pd.read_csv('Validate.csv')
y_pred_knn = model.predict(x_val)
val_raw["Pred"] = y_pred_knn
val_raw.to_csv('Validate_LR.csv', index=False)
train_raw = pd.read_csv('Train.csv')
pred_train = model.predict(x_train)
train_raw["Pred"] = pred_train
train_raw.to_csv('Train_LR.csv', index=False)