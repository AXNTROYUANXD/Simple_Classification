import pandas as pd

FILENAME = "Train.csv"
# FILENAME = "Test.csv"
# FILENAME = "Validate.csv"

df = pd.read_csv(FILENAME, index_col=0)

df['Married'].fillna(0, inplace=True)

df['Gender'].fillna(2, inplace=True)

df['Age'].fillna(-1, inplace=True)

df['Graduate'].fillna(2, inplace=True)

df.loc[df['Profession'].notna(), 'Profession'] = df.loc[df['Profession'].notna(), 'Profession'].replace(
        {
            'Artist': 1,
            'Doctor': 2,
            'Engineer': 3,
            'Entertainment': 4,
            'Executive': 5,
            'Healthcare': 6,
            'Homemaker': 7,
            'Lawyer': 8,
            'Marketing': 9
        })
df['Profession'].fillna(0, inplace=True)

# df = pd.get_dummies(df, columns=['Profession'], dummy_na=True, prefix='Profession', dtype=int)

df['Years_of_Working '].fillna(-1, inplace=True)

df.loc[df['Spending_Score'].notna(), 'Spending_Score'] = df.loc[df['Spending_Score'].notna(), 'Spending_Score'].replace(
        {
            'Low': 1,
            'Average': 2,
            'High': 3
        })
df['Spending_Score'].fillna(0, inplace=True)

# df = pd.get_dummies(df, columns=['Spending_Score'], dummy_na=True, prefix='Spending_Score', dtype=int)

df['Family_Members'].fillna(0, inplace=True)

df.loc[df['Category'].notna(), 'Category'] = df.loc[df['Category'].notna(), 'Category'].replace(
        {
            'Cat_1': 1,
            'Cat_2': 2,
            'Cat_3': 3,
            'Cat_4': 4,
            'Cat_5': 5,
            'Cat_6': 6,
            'Cat_7': 7
        })
df['Category'].fillna(0, inplace=True)

# df = pd.get_dummies(df, columns=['Category'], dummy_na=True, prefix='Category', dtype=int)

# col_name = 'Class(Target)'
# target = df.pop(col_name)
# df = df.assign(col_name=target)

df.to_csv("Processed_" + FILENAME, index=False)