import pandas as pd
import numpy as np
import pickle

# import data
data = pd.read_csv('data/titanic-dataset.csv')

# drop missing values
df = data.copy().dropna()

# data conversion
df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(float)

# data encoding
# Sex
sex_d = {}
i = 0
unique_vals = df['Sex'].unique()
for w in unique_vals:
    sex_d[w] = i
    i+=1
df['Sex'] = df['Sex'].apply(lambda x: sex_d[x])

# Embarked
em_d = {}
i = 0
unique_vals = df['Embarked'].unique()
for w in unique_vals:
    em_d[w] = i
    i+=1
df['Embarked'] = df['Embarked'].apply(lambda x: em_d[x])

# Cabin
cab_d = {}
i = 0
unique_vals = df['Cabin'].unique()
for w in unique_vals:
    cab_d[w] = i
    i+=1
df['Cabin'] = df['Cabin'].apply(lambda x: cab_d[x])

# drop unnecessary columns
df = df.drop(columns=['Ticket']).copy()

# save data 
df.to_csv('data/titanic-dataset-processed.csv', index=False)

# save pipeline
with open ('data/pipeline-1.pkl', 'wb') as f:
    pickle.dump(df, f)