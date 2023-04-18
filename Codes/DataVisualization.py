import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('Train.csv')

# Calculate the percentage of missing values for each column
missing_values_percentage = (df_train.isnull().sum() / df_train.shape[0]) * 100
# Format the result with decimal places
missing_values_percentage = missing_values_percentage.round(1)
print(missing_values_percentage)

# Segmentation based on Gender
sns.countplot(x='Class(Target)', hue='Gender', data=df_train)
plt.title("Class Distribution based on Gender")
plt.show()

# Age based on Segmentation
sns.boxplot(x='Class(Target)', y='Age', data=df_train)
plt.title("Class Distribution based on Age")
plt.show()

# Work Experience vs Spending Score based on Segmentation
sns.boxplot(x='Years_of_Working ', y='Spending_Score', hue='Class(Target)', data=df_train)
plt.title("Work Experience vs Spending Score based on Class")
plt.show()

# Profession based on Segmentation
sns.violinplot(x='Class(Target)', y='Age', hue='Profession', data=df_train)
plt.title("Profession Distribution based on Age")
plt.show()

# Family Size based on Segmentation
df_train['Family_Members'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Family Members Distribution")
plt.show()

# Plot the distribution of 'Age'
plt.figure(figsize=(10,5))
sns.histplot(df_train['Age'], kde=True)
plt.title("Distribution of 'Age'")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Plot the distribution of 'Work_Experience'
plt.figure(figsize=(10,5))
sns.histplot(df_train['Years_of_Working '], kde=True)
plt.title("Distribution of 'Years_of_Working'")
plt.xlabel("Years_of_Working")
plt.ylabel("Frequency")
plt.show()

# Plot the distribution of 'Family_Size'
plt.figure(figsize=(10,5))
sns.histplot(df_train['Family_Members'], kde=True)
plt.title("Distribution of 'Family_Members'")
plt.xlabel("Family_Members")
plt.ylabel("Frequency")
plt.show()

# How many is in each category
cd = df_train.groupby(['Class(Target)'])['ID'].count().reset_index()
cd['cnt_IDS'] = cd['ID']
del cd['ID']
df_train = df_train.merge(cd, on='Class(Target)', how='left')

df_train_kmeans = df_train.drop(['Class(Target)', 'ID'], axis=1)


# Convert the categorical columns to Label encoded columns
from sklearn.preprocessing import  LabelEncoder
encoder = LabelEncoder()
df_train_kmeans['Profession'] = encoder.fit_transform(df_train_kmeans['Profession'])
df_train_kmeans['Spending_Score'] = encoder.fit_transform(df_train_kmeans['Spending_Score'])
df_train_kmeans['Category'] = encoder.fit_transform(df_train_kmeans['Category'])

# Create a correlation matrix
corr = df_train_kmeans.corr()
# Create a heatmap from the correlation matrix
sns.heatmap(corr, annot=True)
# Show the heatmap
plt.show()