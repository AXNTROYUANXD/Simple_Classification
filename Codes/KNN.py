import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier

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

iteration = 100
error_rate = []
acc = []
scores = {}

for i in range(1, iteration):
    model_knn = KNeighborsClassifier(n_neighbors=i)
    model_knn.fit(x_train, y_train)
    y_pred_knn = model_knn.predict(x_val)
    error_rate.append(np.mean(y_pred_knn != y_val))
    scores[i] = metrics.accuracy_score(y_val, y_pred_knn)
    acc.append(metrics.accuracy_score(y_val, y_pred_knn))



plt.figure(figsize=(10,6))
plt.plot(range(1,iteration), error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

plt.figure(figsize=(10,6))
plt.plot(range(1,iteration),acc,color = 'blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

# TODO HERE
model_knn = KNeighborsClassifier(n_neighbors=18)
model_knn.fit(x_train, y_train)

y_pred_knn = model_knn.predict(x_val)

"""
Accuracy :  0.43966323666978485 (on validation set)
"""
print("Accuracy : ", accuracy_score(y_val, y_pred_knn))
cr = classification_report(y_val, y_pred_knn)
print("\t\tClassification Report\n" + "--"*28 + "\n", cr)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_val, y_pred_knn), display_labels=['A', 'B', 'C', 'D'])
cm_display.plot()
plt.show()

y_pred_knn = model_knn.predict(x_test)

print(y_pred_knn)
test_df['Class(Target)'] = y_pred_knn
test_df.to_csv("Test_KNN.csv", index=False)

test_raw = pd.read_csv('Test.csv')
test_raw["Class(Target)"] = y_pred_knn
test_raw.to_csv('Test_KNN.csv', index=False)

val_raw = pd.read_csv('Validate.csv')
y_pred_knn = model_knn.predict(x_val)
val_raw["Pred"] = y_pred_knn
val_raw.to_csv('Validate_KNN.csv', index=False)
train_raw = pd.read_csv('Train.csv')
pred_train = model_knn.predict(x_train)
train_raw["Pred"] = pred_train
train_raw.to_csv('Train_KNN.csv', index=False)