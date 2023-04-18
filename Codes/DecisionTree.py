import pandas as pd


from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


data = pd.read_csv('Processed_Train(1).csv')
X_train = data.drop('Class(Target)', axis=1)
y_train = data['Class(Target)']

test_data = pd.read_csv('Processed_Validate(1).csv')
X_test = test_data.drop('Class(Target)', axis=1)
y_test = test_data['Class(Target)']

# Decision tree
"""
Accuracy: 0.46960
"""
model = DecisionTreeClassifier(max_depth=4, criterion='entropy').fit(X_train, y_train)



predictions = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.5f}")

# Generate confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, predictions), display_labels=['A', 'B', 'C', 'D'])
cm_display.plot()
plt.show()

true_test_data = pd.read_csv('Processed_Test(1).csv')
X_true_test = true_test_data.drop('Class(Target)', axis=1)
pred = model.predict(X_true_test)

test_raw = pd.read_csv('Test.csv')
test_raw["Class(Target)"] = pred
test_raw.to_csv('Test_DT.csv', index=False)

val_raw = pd.read_csv('Validate.csv')
val_raw["Pred"] = predictions
val_raw.to_csv('Validate_DT.csv', index=False)
train_raw = pd.read_csv('Train.csv')
pred_train = model.predict(X_train)
train_raw["Pred"] = pred_train
train_raw.to_csv('Train_DT.csv', index=False)